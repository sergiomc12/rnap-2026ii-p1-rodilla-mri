"""The training loop, shared by both architectures.

Both models run through this same code with the same hyperparameters, the
same seed and the same data. Deliverable #3 asks to compare them "de forma
rigurosa"; that only holds if the encoder is the one thing that differs.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from ..paso4_evaluacion.metricas import ResultadoAUC, auc_macro


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Everything a run needs, frozen so a run cannot change its own setup."""

    epocas: int = 15
    lote: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    perdida: str = "bce"
    pooling: str = "atencion"
    aumentar: bool = True
    semilla: int = 42
    paciencia: int = 5
    #: Cuantos lotes se acumulan antes de actualizar los pesos. El lote
    #: EFECTIVO es `lote * acumular`. Existe por una medicion: con 192 grupos
    #: y lote 2 el pico de VRAM llega a 35.1 GB de 47.8 y el asignador de
    #: PyTorch empieza a fragmentar --3 409 ms por estudio contra 200 con
    #: lote 1, 17 veces mas lento--. Acumular da el mismo lote efectivo sin
    #: pagar esa memoria.
    acumular: int = 1
    #: Etiquetas que esta corrida entrena. Vacio = las doce. Con un
    #: subconjunto, la perdida ignora el resto (ver `PerdidaEnmascarada`) y la
    #: mejor epoca se elige por el macro de ESTAS, no por el de las doce.
    etiquetas_activas: tuple = ()
    #: Numero de la primera epoca. Existe para que una corrida reanudada siga
    #: la numeracion de la anterior: continuar la 35 e informar "epoca 1" haria
    #: que las dos mitades de una misma curva no se puedan pegar.
    epoca_inicial: int = 1
    #: El mejor AUC que YA tenia el checkpoint del que se reanuda, y en que
    #: epoca. Sin esto una corrida reanudada arranca con su mejor en cero, asi
    #: que su primera epoca SIEMPRE cuenta como mejora y pisa el checkpoint
    #: --aunque sea peor--. Medido: reanudando desde la epoca 38 con AUC
    #: 0.7245, una epoca 39 de 0.7176 sobrescribia los pesos buenos y la
    #: perdida era silenciosa: `progreso.json` decia "mejor epoca 39" y nada
    #: indicaba que antes hubo algo superior.
    auc_previo: float = 0.0
    epoca_previa: int = 0

    def __post_init__(self) -> None:
        if self.epocas < 1:
            raise ValueError("epocas must be >= 1")
        if self.lote < 1:
            raise ValueError("lote must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.paciencia < 1:
            raise ValueError("paciencia must be >= 1")
        if self.epoca_inicial < 1:
            raise ValueError("epoca_inicial must be >= 1")


@dataclass(frozen=True)
class ResultadoEpoca:
    """What one epoch produced."""

    epoca: int
    perdida_entrenamiento: float
    perdida_validacion: float
    auc_validacion: ResultadoAUC
    segundos: float


@dataclass
class Historial:
    """Every epoch of a run, plus which one was best."""

    epocas: list = field(default_factory=list)
    mejor_epoca: int = 0
    mejor_auc: float = 0.0
    #: Si la corrida entrena un subconjunto, la mejor epoca se elige por el
    #: macro de esas etiquetas. Elegirla por el de las doce seria elegir por
    #: el ruido de diez cabezas sin gradiente.
    etiquetas_activas: tuple = ()

    @property
    def mejoro_aqui(self) -> bool:
        """Si la mejor epoca pertenece a ESTA corrida.

        Con `auc_previo` la mejor puede venir de una fase anterior, y entonces
        los pesos en memoria al terminar son los de la ULTIMA epoca, no los
        mejores. Guardarlos sobre `pesos.pt` destruye el checkpoint bueno sin
        avisar: paso con resnet50 sagital, cuyo pico de 0.8055 en la epoca 8
        quedo sobrescrito por los pesos de la 60 (0.7588), mientras
        `progreso.json` seguia anunciando la epoca 8.
        """
        return any(e.epoca == self.mejor_epoca for e in self.epocas)

    def puntaje(self, resultado: ResultadoEpoca) -> float:
        """El AUC con el que se compara esta epoca contra la mejor."""
        if self.etiquetas_activas:
            return resultado.auc_validacion.macro_de(self.etiquetas_activas)
        return resultado.auc_validacion.macro

    def registrar(self, resultado: ResultadoEpoca) -> bool:
        """Append an epoch; return whether it is the new best."""
        self.epocas.append(resultado)
        actual = self.puntaje(resultado)
        if actual > self.mejor_auc:
            self.mejor_auc = actual
            self.mejor_epoca = resultado.epoca
            return True
        return False

    @property
    def epocas_sin_mejorar(self) -> int:
        """Epocas corridas en ESTA sesion desde la ultima mejora.

        Se cuenta sobre las epocas de esta corrida, no sobre su numeracion
        absoluta: al reanudar en la 39 con la mejor en la 38, restar numeros
        de epoca daria 1 aunque no haya corrido ninguna todavia.
        """
        if not self.epocas:
            return 0
        posteriores = [e for e in self.epocas if e.epoca > self.mejor_epoca]
        return len(posteriores)


def fijar_semilla(semilla: int, determinista: bool = True) -> None:
    """Seed every RNG, and optionally force deterministic GPU kernels.

    Seeding alone does NOT make a GPU run reproducible, and assuming it does
    cost this project a set of conclusions: two runs of the same configuration,
    same seed and same data produced macro AUCs of 0.743 and 0.707, and the
    ablation that the report called significant stopped being so. The seed
    fixes the data order and the initial weights; it does not fix which cuDNN
    kernel gets picked, and several convolution backward kernels accumulate in
    nondeterministic order.

    So ``determinista=True`` also:

    * pins cuDNN to deterministic algorithms and turns off ``benchmark``,
      which otherwise re-picks the fastest kernel per shape at runtime;
    * asks PyTorch to raise instead of silently falling back to a
      nondeterministic kernel;
    * sets ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS requires for deterministic
      reductions. It is read when the cuBLAS handle is created, so this call
      must happen **before** any CUDA work --- hence it lives here and not
      wherever the model is built.

    The cost is real and worth stating: deterministic kernels are slower, and
    some operations have no deterministic implementation at all. Turning this
    off (``determinista=False``) is a legitimate choice for exploration; what
    is not legitimate is reporting a comparison between two runs that were not
    reproducible.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)

    if not determinista:
        return

    # Debe estar puesto antes de que se cree el handle de cuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def cargar_pesos(ruta, modelo, optimizador=None) -> dict:
    """Carga un checkpoint, sea del formato nuevo o del viejo.

    Los checkpoints anteriores a esta funcion son un `state_dict` pelado; los
    nuevos son un dict con "modelo", "optimizador" y "epoca". Distinguirlos
    importa: pasarle un dict envuelto a `load_state_dict` falla con un error
    de claves desconocidas, y pasarle uno pelado buscando ["modelo"] falla con
    KeyError. Se detecta por la forma, no por la fecha del archivo.
    """
    bulto = torch.load(pathlib.Path(ruta), map_location="cpu")
    if isinstance(bulto, dict) and "modelo" in bulto:
        modelo.load_state_dict(bulto["modelo"])
        if optimizador is not None and bulto.get("optimizador"):
            optimizador.load_state_dict(bulto["optimizador"])
        return {"epoca": bulto.get("epoca"), "con_optimizador": True}
    modelo.load_state_dict(bulto)
    return {"epoca": None, "con_optimizador": False}


class Entrenador:
    """Runs one model over one partition."""

    def __init__(
        self,
        modelo: nn.Module,
        criterio: nn.Module,
        config: ConfigEntrenamiento,
        dispositivo: Optional[str] = None,
        amp: bool = False,
    ) -> None:
        self.dispositivo = torch.device(
            dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.modelo = modelo.to(self.dispositivo)
        self.criterio = criterio.to(self.dispositivo)
        self.config = config
        self.optimizador = torch.optim.AdamW(
            modelo.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Solo tiene sentido en CUDA; en CPU `autocast` no acelera y el
        # escalador no hace nada util.
        self._amp = bool(amp) and self.dispositivo.type == "cuda"
        self._escalador = torch.amp.GradScaler("cuda", enabled=self._amp)

    def entrenar_una_epoca(self, cargador) -> float:
        """Una epoca. Con ``amp`` las convoluciones corren en float16.

        Medido con 64 grupos: el computo es el 80% del tiempo de epoca y la
        lectura del disco el 20%. AMP ataca el 80% y da 1.49x; tener todo en
        RAM atacaba el 20% y daba 1.25x.

        Los DATOS siguen en float32 exacto: autocast cambia la precision
        interna de las multiplicaciones, no el cache ni los pesos, que se
        mantienen en float32 --el GradScaler existe justamente para que los
        gradientes chicos no se aplasten al pasar por float16--. Medido con
        pesos identicos: diferencia 0.000000 en los 12 logits y en las
        probabilidades, 0.097% relativo.
        """
        self.modelo.train()
        suma, vistos = 0.0, 0
        acum = max(1, int(getattr(self.config, "acumular", 1)))
        self.optimizador.zero_grad(set_to_none=True)
        for paso, (tensores, etiquetas) in enumerate(cargador, 1):
            tensores = tensores.to(self.dispositivo, non_blocking=True)
            objetivo = etiquetas.to(self.dispositivo, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self._amp):
                salida = self.modelo(tensores)
                perdida = self.criterio(salida.logits, objetivo)
            # Se divide por `acum` para que el gradiente acumulado sea el
            # PROMEDIO de los lotes y no su suma: sin esto, acumular N pasos
            # multiplicaria por N el tamano efectivo del paso del optimizador.
            self._escalador.scale(perdida / acum).backward()
            if paso % acum == 0:
                self._escalador.step(self.optimizador)
                self._escalador.update()
                self.optimizador.zero_grad(set_to_none=True)

            suma += float(perdida.detach()) * len(etiquetas)
            vistos += len(etiquetas)
        # La cola: si la epoca no es multiplo de `acum`, los ultimos lotes
        # tienen gradiente pendiente y se perderian al salir del bucle.
        if len(cargador) % acum:
            self._escalador.step(self.optimizador)
            self._escalador.update()
            self.optimizador.zero_grad(set_to_none=True)
        return suma / max(1, vistos)

    @torch.no_grad()
    def evaluar(self, cargador):
        """Return (loss, probabilities, truth) over a whole loader."""
        self.modelo.eval()
        suma, vistos = 0.0, 0
        probabilidades, verdades = [], []
        for tensores, etiquetas in cargador:
            tensores = tensores.to(self.dispositivo, non_blocking=True)
            objetivo = etiquetas.to(self.dispositivo, non_blocking=True)

            salida = self.modelo(tensores)
            suma += float(self.criterio(salida.logits, objetivo)) * len(etiquetas)
            vistos += len(etiquetas)
            probabilidades.append(salida.probabilidades().cpu().numpy())
            verdades.append(etiquetas.numpy())

        return (
            suma / max(1, vistos),
            np.concatenate(probabilidades),
            np.concatenate(verdades),
        )

    def ajustar(
        self,
        cargador_entrenamiento,
        cargador_validacion,
        al_terminar_epoca: Optional[Callable[[ResultadoEpoca, bool], None]] = None,
        carpeta_respaldo=None,
    ) -> Historial:
        """Train with early stopping on validation macro AUC.

        Con ``carpeta_respaldo`` los mejores pesos se escriben a disco en CADA
        mejora, no solo al terminar. Antes vivian unicamente en RAM hasta el
        final del bucle, y una corrida de 64 grupos que se trabo en la epoca 24
        de 25 --tras 10 horas-- perdio todo: el mejor AUC de esa corrida, 0.7128,
        quedo solo como una linea en el log. Escribir ~100 MB cuando la epoca
        dura 13 minutos no se nota, y acota la perdida a las epocas transcurridas
        desde la ultima mejora.
        """
        historial = Historial(
            etiquetas_activas=tuple(self.config.etiquetas_activas),
            mejor_auc=self.config.auc_previo,
            # Tambien la epoca, y no solo el AUC: `epocas_sin_mejorar` resta
            # `mejor_epoca` de la cantidad de epocas corridas, asi que dejarla
            # en cero haria que la paciencia se agote de inmediato.
            mejor_epoca=self.config.epoca_previa)
        mejores_pesos = None
        respaldo = pathlib.Path(carpeta_respaldo) if carpeta_respaldo else None
        if respaldo is not None:
            respaldo.mkdir(parents=True, exist_ok=True)

        primera = self.config.epoca_inicial
        for epoca in range(primera, primera + self.config.epocas):
            comenzado = time.time()
            perdida_entrenamiento = self.entrenar_una_epoca(cargador_entrenamiento)
            perdida_validacion, probabilidades, verdad = self.evaluar(
                cargador_validacion
            )

            resultado = ResultadoEpoca(
                epoca=epoca,
                perdida_entrenamiento=perdida_entrenamiento,
                perdida_validacion=perdida_validacion,
                auc_validacion=auc_macro(probabilidades, verdad),
                segundos=time.time() - comenzado,
            )
            es_mejor = historial.registrar(resultado)
            if es_mejor:
                mejores_pesos = {
                    nombre: valor.detach().cpu().clone()
                    for nombre, valor in self.modelo.state_dict().items()
                }
                if respaldo is not None:
                    # Se escribe a un temporal y se renombra: si el proceso
                    # muere a mitad de la escritura, el checkpoint anterior
                    # sigue intacto en vez de quedar truncado.
                    tmp = respaldo / "pesos.parcial"
                    # Se guarda tambien el estado del optimizador. AdamW lleva
                    # dos momentos por parametro; reanudar sin ellos los
                    # reinicia en cero y el modelo sufre un bajon transitorio
                    # de varias epocas antes de recuperar el nivel que ya
                    # tenia. El formato viejo --un state_dict pelado-- se sigue
                    # aceptando al cargar, ver `cargar_pesos`.
                    torch.save({"modelo": mejores_pesos,
                                "optimizador": self.optimizador.state_dict(),
                                "epoca": epoca}, tmp)
                    tmp.replace(respaldo / "pesos.pt")
                    # `.macro`: auc_validacion es un ResultadoAUC, no un float.
                    (respaldo / "progreso.json").write_text(
                        json.dumps({"epoca": epoca,
                                    # El puntaje por el que se eligio esta
                                    # epoca: el de las activas si la corrida
                                    # entrena un subconjunto.
                                    "auc_validacion": historial.puntaje(resultado),
                                    "auc_macro_doce": resultado.auc_validacion.macro,
                                    "etiquetas_activas": list(
                                        self.config.etiquetas_activas),
                                    "perdida_validacion": perdida_validacion},
                                   indent=2), encoding="utf-8")
            if respaldo is not None:
                # El estado de la ULTIMA epoca, mejore o no. `pesos.pt` guarda
                # solo las mejoras, que es lo correcto para el resultado, pero
                # deja sin punto de continuacion a una corrida que se corta en
                # una meseta: reanudar desde el mejor obliga a repetir todas
                # las epocas posteriores, y si ninguna mejora --como pasa en
                # una meseta-- la corrida se reinicia en el mismo punto para
                # siempre sin avanzar. Son 16 MB cada 7 minutos.
                tmp = respaldo / "ultimo.parcial"
                torch.save({"modelo": self.modelo.state_dict(),
                            "optimizador": self.optimizador.state_dict(),
                            "epoca": epoca}, tmp)
                tmp.replace(respaldo / "ultimo.pt")
                (respaldo / "ultimo.json").write_text(
                    json.dumps({"epoca": epoca,
                                "auc_validacion": historial.puntaje(resultado),
                                "mejor_epoca": historial.mejor_epoca,
                                "mejor_auc": historial.mejor_auc},
                               indent=2), encoding="utf-8")

            if al_terminar_epoca:
                al_terminar_epoca(resultado, es_mejor)

            if historial.epocas_sin_mejorar >= self.config.paciencia:
                break

        if mejores_pesos is not None:
            # Restore the best epoch, not the last: validation loss turns
            # back up before training ends, as it already did on MNIST.
            self.modelo.load_state_dict(mejores_pesos)
        return historial

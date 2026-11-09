"""neuralcello: inference for the neural cello models (codec, latent flow, note-conditioned curve model, symbolic composer).

    import neuralcello as nc
    m = nc.load()                                  # device=None -> CUDA if available, else CPU; models load lazily from ../models
    pm = m.compose(style="folk", key="D minor", meter="3/4", bars=24, bpm=58, tau=0.6, seed=0)     # -> pretty_midi.PrettyMIDI
    y = m.perform(pm, vib=0.6, seed=0)             # -> float32 audio at 24 kHz
    z = m.encode(wav, sr); y2 = m.decode(z)        # audio <-> 24-channel latent at 25 Hz

The module-level encode / decode / compose / perform use one shared default instance."""
from .pipeline import NeuralCello, MODELS
from .codec import SR

__all__ = ["load", "encode", "decode", "compose", "perform", "NeuralCello", "SR"]
_default = None


def load(device=None, models=MODELS, tf32=True):
    """-> NeuralCello holding codec, latent model, curve model and composer (each loaded on first use). tf32: enable TF32 matmuls on CUDA, as in training."""
    global _default
    _default = NeuralCello(device, models, tf32); return _default


def _get(): return _default or load()
def encode(wav, sr=SR): return _get().encode(wav, sr)
def decode(latent, seed=0): return _get().decode(latent, seed)
def compose(**kw): return _get().compose(**kw)
def perform(midi, **kw): return _get().perform(midi, **kw)

"""fungalplm -- a small protein language model trained on the fungal kingdom."""
import sys

from .api import FungalPLM, read_fasta
from .model import Vocab, Transformer, load_checkpoint

__version__ = "0.1.0"
__all__ = ["FungalPLM", "read_fasta", "Vocab", "Transformer", "load_checkpoint"]

_BANNER = r"""
         ___..._
    _,--'       "`-.
  ,'.  .  Cheers!   \
,/:. .     .       .'
|;..  .      _..--'
`--:...-,-'""\
        |:.  `.
        l;.   l
        `|:.   |
         |:.   `.,
        .l;.    j, ,
     `. \`;:.   //,/
      .\\)`;,|\'/(
       ` `itz `(,
"""

if sys.stdout and sys.stdout.isatty():   # fun banner for humans in a terminal; silent in scripts/pipes
    print(_BANNER)

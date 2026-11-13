"""Run existing matched-image controls with the n=4 SGKLD paper port.

Example: python tools/check_paper_sgkld.py --steps 100 --anchor-images 4
         --work-dir work_dirs/dior_r_trainval/sgkld_probe_20261103
The inherited CLI also accepts --arms riou sgkld (default: both).
"""
from pathlib import Path
import sys

import check_paper_losses as probe
import mmrotate.models.losses.paper_sgkld_loss  # noqa: F401


def main():
    probe.CONFIGS = dict(riou='riou.py', sgkld='paper_sgkld.py')
    original = probe.digest
    extra = ['mmrotate/models/losses/paper_sgkld_loss.py',
             'tools/check_paper_sgkld.py']

    def digest_with_sgkld(paths):
        values = original(list(dict.fromkeys(list(paths) + extra)))
        # The base harness snapshots its own source list. Include the added
        # objective and this entry point too, before any model construction.
        if '--work-dir' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--work-dir') + 1]).resolve()
            if output.is_relative_to(probe.ROOT / 'work_dirs') and output.exists():
                for name in extra:
                    saved = output / 'snapshot' / name
                    if not saved.exists():
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        saved.write_bytes((probe.ROOT / name).read_bytes())
        return values

    probe.digest = digest_with_sgkld
    probe.main()


if __name__ == '__main__':
    main()

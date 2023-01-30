

from .deformable_detr import build as build_deformable_detr
from .motr import build as build_motr


def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'motr': build_motr,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)


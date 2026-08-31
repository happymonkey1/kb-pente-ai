from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"

SOURCES = [
    NATIVE / "torch" / "bindings.cpp",
    NATIVE / "src" / "position.cpp",
    NATIVE / "src" / "position_hash.cpp",
    NATIVE / "src" / "rules.cpp",
    NATIVE / "src" / "game.cpp",
    NATIVE / "src" / "parallel" / "worker_pool.cpp",
    NATIVE / "src" / "mcts" / "inference_workspace.cpp",
    NATIVE / "src" / "mcts" / "tree_arena.cpp",
    NATIVE / "src" / "mcts" / "tree.cpp",
    NATIVE / "src" / "mcts" / "search_session.cpp",
    NATIVE / "src" / "mcts" / "search_batch.cpp",
]


setup(
    name="kb_pente_native",
    ext_modules=[
        CppExtension(
            name="kb_pente_native",
            sources=[str(source) for source in SOURCES],
            include_dirs=[str(NATIVE / "include")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-Wall",
                    "-Wextra",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

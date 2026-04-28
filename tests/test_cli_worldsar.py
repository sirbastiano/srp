import sys

import pytest

from sarpyx.cli.worldsar import create_parser, main


WORLDSAR_DEFAULT_FIELDS = (
    'output_dir',
    'cuts_outdir',
    'gpt_path',
    'grid_path',
    'db_dir',
    'gpt_memory',
    'gpt_parallelism',
    'gpt_timeout',
    'snap_userdir',
    'orbit_type',
)


def test_create_parser_formats_help() -> None:
    parser = create_parser()
    help_text = parser.format_help()

    assert '--input' in help_text
    assert '--orbit-continue-on-fail' in help_text


def test_create_parser_parses_worldsar_arguments() -> None:
    parser = create_parser()

    args = parser.parse_args(
        [
            '--input',
            '/tmp/product.SAFE',
            '--output',
            '/tmp/output',
            '--cuts-outdir',
            '/tmp/cuts',
            '--db-dir',
            '/tmp/db',
            '--h5-to-zarr-only',
            '--zarr-chunk-size',
            '64',
            '64',
            '--orbit-continue-on-fail',
        ]
    )

    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.h5_to_zarr_only is True
    assert args.zarr_chunk_size == [64, 64]
    assert args.orbit_continue_on_fail is True


def test_create_parser_defaults_are_stable() -> None:
    args = create_parser().parse_args(['--input', '/tmp/product.SAFE'])

    for field in WORLDSAR_DEFAULT_FIELDS:
        assert '/shared/home/vmarsocci' not in str(getattr(args, field))


def test_main_dispatches_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from sarpyx.cli import worldsar as worldsar_module

    calls = []

    monkeypatch.setattr(worldsar_module, 'run', lambda args: calls.append(args) or 7)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'sarpyx',
            '--input',
            '/tmp/product.SAFE',
            '--output',
            '/tmp/output',
            '--gpt-path',
            '/tmp/gpt',
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 7
    assert len(calls) == 1
    assert calls[0].product_path == '/tmp/product.SAFE'
    assert calls[0].output_dir == '/tmp/output'
    assert calls[0].gpt_path == '/tmp/gpt'

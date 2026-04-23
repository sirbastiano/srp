from sarpyx.cli.main import create_main_parser


def test_create_main_parser_formats_help_with_worldsar_subcommand() -> None:
    parser = create_main_parser()
    help_text = parser.format_help()

    assert 'worldsar' in help_text
    assert '--version' in help_text


def test_create_main_parser_includes_worldsar_subcommand() -> None:
    parser = create_main_parser()

    args = parser.parse_args(
        [
            'worldsar',
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

    assert args.command == 'worldsar'
    assert args.product_path == '/tmp/product.SAFE'
    assert args.output_dir == '/tmp/output'
    assert args.cuts_outdir == '/tmp/cuts'
    assert args.db_dir == '/tmp/db'
    assert args.h5_to_zarr_only is True
    assert args.zarr_chunk_size == [64, 64]
    assert args.orbit_continue_on_fail is True

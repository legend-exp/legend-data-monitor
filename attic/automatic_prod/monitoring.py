import argparse

import legend_data_monitor


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    func1_parser = subparsers.add_parser(
        "summary_files", help="Run function for creating summary HDF and YAML files."
    )
    func1_parser.add_argument(
        "--path", help="Path to the folder containing the monitoring HDF files."
    )
    func1_parser.add_argument("--period", help="Period to inspect.")
    func1_parser.add_argument("--run", help="Run to inspect.")

    func2_parser = subparsers.add_parser(
        "plot", help="Run function for creating summary plots."
    )
    func2_parser.add_argument(
        "--public_data",
        help="Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).",
        default="/data2/public/prodenv/prod-blind/ref-v1.0.1",
    )
    func2_parser.add_argument(
        "--hdf_files",
        help="Path to generated monitoring hdf files.",
    )
    func2_parser.add_argument(
        "--output", default="removal_new_keys", help="Path to output folder."
    )
    func2_parser.add_argument("--start", help="First timestamp of the inspected range.")
    func2_parser.add_argument("--p", help="Period to inspect.")
    func2_parser.add_argument(
        "--avail_runs",
        nargs="+",
        type=str,
        help="Available runs to inspect for a given period.",
    )
    func2_parser.add_argument("--current_run", type=str, help="Run under inspection.")
    func2_parser.add_argument(
        "--partition",
        default=False,
        help="False if not partition data; default: False.",
    )
    func2_parser.add_argument(
        "--zoom", default=False, help="True to zoom over y axis; default: False."
    )
    func2_parser.add_argument(
        "--quad_res",
        default=False,
        help="True if you want to plot the quadratic resolution too; default: False.",
    )
    func2_parser.add_argument(
        "--pswd_email",
        default=None,
        help="Password to access the legend.data.monitoring@gmail.com account for sending alert messages.",
    )
    func2_parser.add_argument(
        "--escale",
        default=2039.0,
        type=float,
        help="Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).",
    )
    func2_parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False.",
    )
    func2_parser.add_argument(
        "--last_checked",
        help="Timestamp of the last check.",
    )

    func3_parser = subparsers.add_parser(
        "calib_psd",
        help="Plot and raise warning for PSD stability in calibration runs.",
    )
    func3_parser.add_argument(
        "--public_data",
        help="Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).",
        default="/data2/public/prodenv/prod-blind/ref-v1.0.1",
    )
    func3_parser.add_argument(
        "--output", default="removal_new_keys", help="Path to output folder."
    )
    func3_parser.add_argument("--p", help="Period to inspect.")
    func3_parser.add_argument("--current_run", type=str, help="Run under inspection.")
    func3_parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False.",
    )

    args = parser.parse_args()

    if args.command == "summary_files":
        legend_data_monitor.monitoring.build_new_files(args.path, args.period, args.run)

    elif args.command == "plot":
        auto_dir_path = args.public_data
        phy_mtg_data = args.hdf_files
        output_folder = args.output
        start_key = args.start
        period = args.p
        runs = args.avail_runs
        current_run = args.current_run
        pswd_email = args.pswd_email
        save_pdf = args.pdf
        escale_val = args.escale
        last_checked = args.last_checked
        partition = args.partition
        quadratic = args.quad_res
        zoom = args.zoom

        legend_data_monitor.monitoring.plot_time_series(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            runs,
            current_run,
            pswd_email,
            save_pdf,
            escale_val,
            last_checked,
            partition,
            quadratic,
            zoom,
        )

    elif args.command == "calib_psd":
        auto_dir_path = args.public_data
        output_folder = args.output
        period = args.p
        save_pdf = args.pdf
        current_run = args.current_run

        legend_data_monitor.calibration.check_psd(
            auto_dir_path, output_folder, period, current_run, save_pdf
        )


if __name__ == "__main__":
    main()

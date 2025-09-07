import os

from legend_data_monitor.utils import get_output_plot_path


def test_get_output_plot_path(tmp_path):
    input_path = tmp_path / "plt" / "hit" / "phy" / "myplot"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("dummy content")

    result = get_output_plot_path(str(input_path), "pdf")

    assert result.endswith("myplot.pdf")
    assert "tmp/mtg/" in result
    assert os.path.isdir(os.path.dirname(result))

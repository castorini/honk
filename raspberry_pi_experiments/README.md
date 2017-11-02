# Model Power Consumption Benchmark Guide

This documents how we ran the power consumption experiments for the models on the Raspberry Pi. We use a [Watts Up Pro](https://www.vernier.com/products/sensors/wu-pro/) meter to measure the power. The results are summarized in our [paper](https://arxiv.org/abs/1711.00333) on arXiv.

## Experiment Setup
1. Download and install the usb serial from `http://www.ftdichip.com/Drivers/VCP.htm`.
2. Plug the Raspberry Pi into the Watts Up Pro and adjust the mode to be current Watt mode.
3. Connect Wattsup Pro with your laptop using usb cable.
4. After the Raspberry Pi (assume the wifi is connectable) (TODO(tuzhucheng): add intruction to setup wifi) is fully started, connect your laptop to the wifi provided by the Raspberry Pi.
5. Kick start the `wattsup_server.py` on your laptop, and record the printed out `wattsup_server ip`.
6. Double check the model name and the keywords in the `power_consumption_benchmark.py` on Raspberry Pi. (also make sure that the idle watt read is greater or equal to the one shown in the script)
7. Kick start `power_consumption_benchmark.py` on Raspberry Pi with the `wattsup_server ip` and the port your `wattsup_server` is running on and get read from stdout.

## Analysis
You can save the output of the `power_consumption_benchmark.py` script into a file. `experiment_output_e2e.txt` is our end-to-end experiment result including both preprocessing and inference while `experiment_output_preprocessing.txt` is our experiment result with preprocessing only.

We do some exploratory analysis in a Jupyter Notebook `analysis.ipynb`. Note the dependencies in the notebook (`numpy`, `pandas`, `matplotlib`, `seaborn`) are not in any `requirements*.txt` files so you need to install them to run the commands in the notebook. This notebook also outputs two CSV files for more detailed analysis in R, like p-values and using `ggplot2` for plotting.

The R Markdown file `analysis_plots.Rmd` is used to generate the plots for our paper. It takes as input files outputted by the Jupyter notebook.

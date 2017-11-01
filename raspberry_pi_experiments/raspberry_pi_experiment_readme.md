## Model power consumption benchmark guide
### Experiment Setup
1. Download and install the usb serial from `http://www.ftdichip.com/Drivers/VCP.htm`.
2. Plug the rpi into the Wattsup Pro and adjust the mode to be current watt mode.
3. Connect Wattsup Pro with your laptop using usb cable.
4. After the rpi (assume the wifi is connectable) is fully started, connect your laptop to the wifi provided by the rpi.
5. Kick start the `wattsup_server.py` on your loptop, and record the printed out `wattsup_server ip`.
6. Double check the model name and the keywords in the `power_consumption_benchmark.py` on rpi. (also make sure that the idle watt read is greater or equal to the one shown in the script)
7. Kick start `power_consumption_benchmark.py` on rpi with the `wattsup_server ip` and the port your `wattsup_server` is running on and get read from stdout.

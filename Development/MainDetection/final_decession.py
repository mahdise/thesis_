"""
This script will be used for
the final result about the orientation
of the parcels.

"""
import time
from multiprocessing import Queue
from Development.MainDetection.gui_methods import GuiObject
from Development.MainDetection.data_analysis import data_analysis_for_processes


def final_decision(top_cam_data: Queue, side_cam_data: Queue,data_write_path):
    print("final_decision starting ................... ")

    gui_display = GuiObject()
    while True:
        # Could also delay for calculation
        time.sleep(1)
        data_from_top_cam = top_cam_data.get()
        data_from_side_cam = side_cam_data.get()

        data_analysis_for_processes(data_from_top_cam, data_from_side_cam, data_write_path, gui_display)


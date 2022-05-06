from datetime import datetime

import pandas as pd

from Development.MainDetection.data_read_write import write_data




def data_analysis_for_processes(top_cam_data, side_cam_data, data_write_path, gui_display):
    now = datetime.now().time()  # time object

    file_path_for_orientation_hist = data_write_path["orientation_result"]

    final_result = dict()
    # take maximum num detecting boxes
    try:
        total_boxes = max([int(top_cam_data["total_boxes"]), int(side_cam_data["total_boxes"])])

    except:
        total_boxes = 0

    if total_boxes > 0:
        all_box_number = " "
        all_box_angle = " "
        all_box_arrow_dir = " "
        all_box_arrow_pos = " "
        all_box_ori_type = " "

        for box in range(0, total_boxes):
            label_box = str(box + 1)
            label_box_gui = "Box_" + label_box + " | "
            all_box_number = all_box_number + label_box_gui

            __angle_format = "box_" + label_box + "_angle"
            __angle_type_format = "box_" + label_box + "_angle_type"
            __arrow_top_format_pos = "box_" + label_box + "_arrowTop"
            __arrow_top_format_direction = "box_" + label_box + "_arrowDirection"
            __arrow_side_format_pos = "box_" + label_box + "_arrowSide"
            __arrow_side_format_direction = "box_" + label_box + "_arrowDirection"
            __arrow_top_format_confidence = "box_" + label_box + "_confidence"
            __arrow_side_format_confidence = "box_" + label_box + "_confidence"

            final_result["box_" + label_box] = {
                "rotate_angle": top_cam_data.get(__angle_format) or None,
                "arrow_position": None,
                "arrow_direction": None,
                "orientation_type": None,
                "box_detect_confidence_top_cam": top_cam_data.get(__arrow_top_format_confidence) or None,
                "box_detect_confidence_side_cam": side_cam_data.get(__arrow_side_format_confidence) or None,
                "total_boxes": int(total_boxes),
                "angle_type": top_cam_data.get(__angle_type_format) or None,
                "data_write_time": now

            }
            # get all value
            # print("top cam data : ", top_cam_data)
            arrow_top_pos = top_cam_data.get(__arrow_top_format_pos) or False
            arrow_top_direction = top_cam_data.get(__arrow_top_format_direction) or False

            arrow_side_pos = side_cam_data.get(__arrow_side_format_pos) or False
            arrow_side_direction = side_cam_data.get(__arrow_side_format_direction) or False

            if arrow_top_pos:
                final_result["box_" + label_box]["arrow_position"] = "Top"
                final_result["box_" + label_box]["arrow_direction"] = arrow_top_direction
                final_result["box_" + label_box]["orientation_type"] = "Vertical-" + str(arrow_top_direction)

            else:
                if arrow_side_pos:
                    final_result["box_" + label_box]["arrow_position"] = "Side"
                    final_result["box_" + label_box]["arrow_direction"] = arrow_side_direction

                    type_ori = "Horizontal-"
                    if arrow_side_direction in ["Up", "Down"]:
                        type_ori = "Z-Horizontal-"
                    final_result["box_" + label_box]["orientation_type"] = type_ori + str(arrow_side_direction)

                # extend later for front cam
            # write data for hist
            write_data([label_box,
                        final_result["box_" + label_box]["rotate_angle"],
                        final_result["box_" + label_box]["arrow_position"],
                        final_result["box_" + label_box]["arrow_direction"],
                        final_result["box_" + label_box]["orientation_type"],
                        final_result["box_" + label_box]["box_detect_confidence_top_cam"],
                        final_result["box_" + label_box]["box_detect_confidence_side_cam"],
                        final_result["box_" + label_box]["total_boxes"],
                        final_result["box_" + label_box]["angle_type"],
                        final_result["box_" + label_box]["data_write_time"],
                        ], file_path_for_orientation_hist)

            # update gui table
            try:
                gui_display.update_table(box_umber=str(label_box),
                                         angle=final_result["box_" + label_box]["rotate_angle"],
                                         arrow_dir=final_result["box_" + label_box]["arrow_direction"],
                                         arrow_position=final_result["box_" + label_box]["arrow_position"],
                                         )

            except:
                gui_display.update_table()

            label_gui_angle = str(final_result["box_" + label_box]["rotate_angle"]) + " | "
            all_box_angle = all_box_angle + label_gui_angle

            label_gui_arrow_position = str(final_result["box_" + label_box]["arrow_position"]) + " | "
            all_box_arrow_pos = all_box_arrow_pos + label_gui_arrow_position

            label_gui_arrow_dir = str(final_result["box_" + label_box]["arrow_direction"]) + " | "
            all_box_arrow_dir = all_box_arrow_dir + label_gui_arrow_dir

            label_gui_ori_type = str(final_result["box_" + label_box]["orientation_type"]) + " | "
            all_box_ori_type = all_box_ori_type + label_gui_ori_type

        # add data to gui for final result
        try:
            gui_display.update_final_result(box_umber=all_box_number,
                                            angle=all_box_angle,
                                            arrow_dir=all_box_arrow_dir,
                                            arrow_position=all_box_arrow_pos,
                                            ori_type=all_box_ori_type
                                            )
        except:
            gui_display.update_final_result()
        gui_display.window.update()

    else:
        label_box = "0"

        final_result["box_" + label_box] = {
            "rotate_angle": None,
            "arrow_position": None,
            "arrow_direction": None,
            "orientation_type": None,
            "box_detect_confidence_top_cam": None,
            "box_detect_confidence_side_cam": None,
            "total_boxes": int(0),
            "angle_type": None,
            "data_write_time": now

        }

        write_data([label_box,
                    final_result["box_" + label_box]["rotate_angle"],
                    final_result["box_" + label_box]["arrow_position"],
                    final_result["box_" + label_box]["arrow_direction"],
                    final_result["box_" + label_box]["orientation_type"],
                    final_result["box_" + label_box]["box_detect_confidence_top_cam"],
                    final_result["box_" + label_box]["box_detect_confidence_side_cam"],
                    final_result["box_" + label_box]["total_boxes"],
                    final_result["box_" + label_box]["angle_type"],
                    final_result["box_" + label_box]["data_write_time"],
                    ], file_path_for_orientation_hist)


def data_analysis_for_final_result(get_directory_path):
    """
    Orientation type : Vertical, Horizontal, Z-Horizontal
    :param get_directory_path:
    :return:
    """
    result = {
        "raw_data": None,
        "orientation_type": None,
        "rotation_angle": None
    }

    read_data_pos = pd.read_csv(get_directory_path + "/detection_info_arrow_pos.csv")
    read_data_dir = pd.read_csv(get_directory_path + "/detection_info_arrow_dir.csv")
    read_data_angle = pd.read_csv(get_directory_path + "/detection_info_angle.csv")
    if len(read_data_pos) != 0 and len(read_data_dir) != 0 and len(read_data_angle) != 0:
        latest_data_pos = read_data_pos.head(1)
        latest_data_dir = read_data_dir.head(1)
        latest_data_angle = read_data_angle.head(1)

        if len(latest_data_pos) and len(latest_data_dir) and len(latest_data_dir) > 0:
            arrow_position = latest_data_pos["arrow_position"][0]
            arrow_direction = latest_data_dir["arrow_direction"][0]
            angle_of_box = latest_data_angle["angle_box"][0]

            result["raw_data"] = [angle_of_box, arrow_direction, arrow_position]

            if arrow_position == "top":
                type_ori = "Vertical-" + str(arrow_direction)
                result["orientation_type"] = type_ori

            elif arrow_position in ["side", "front"]:
                type_ori = "Horizontal-"
                if arrow_direction in ["top", "down"]:
                    type_ori = "Z-Horizontal-"
                type_ori = type_ori + str(arrow_direction)
                result["orientation_type"] = type_ori

            result["rotation_angle"] = angle_of_box

    return result

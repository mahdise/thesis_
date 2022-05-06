import pyrealsense2 as rs


def __get_serial_num():
    list_of_camera = list()
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            list_of_camera.append(d.get_info(rs.camera_info.serial_number))
            # print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
    return list_of_camera


if __name__ == '__main__':
    print(__get_serial_num())

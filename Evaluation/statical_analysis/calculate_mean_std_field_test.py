# calculated mean
from Evaluation.utils import read_data
from Evaluation.statical_analysis.calclate_recall_precission import calculate_pre_recall_accuracy, \
    calculate_mse_rmse_r2_score


def calculate_statical_analysis_value(processed_data_frame, processed_data_frame_gt, save_result=True):
    csv_name = "statistical_" + processed_data_frame.rpartition('/')[2]

    data_frame = read_data(processed_data_frame)
    data_frame_gt = read_data(processed_data_frame_gt)

    list_of_data_frame_for_multi_box = list()
    df_box_number_1 = data_frame[data_frame['box_number'] == 1]
    # df_box_number_2 = data_frame[data_frame['box_number'] ==2]
    # df_box_number_3 = data_frame[data_frame['box_number'] ==3]

    list_of_data_frame_for_multi_box.append(df_box_number_1)
    # list_of_data_frame_for_multi_box.append(df_box_number_2)
    # list_of_data_frame_for_multi_box.append(df_box_number_3)
    data_frame_gt.dropna(inplace=True)

    for box_number, data_frame_box in enumerate(list_of_data_frame_for_multi_box):
        csv_name = "statistical_" + str(box_number + 1) + "_" + processed_data_frame.rpartition('/')[2]

        analysis_data = data_frame_box.describe()
        analysis_data = analysis_data.drop(['box_number', 'total_boxes', 'orientation_type_encode'], axis=1)
        # calculate_tp_fp_fn_tn()

        analysis_data["parameter"] = analysis_data.index

        ## For Recall #####

        parameter = ["pricission", "recall"]
        df2 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "pricission"}

        df3 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "recall"}

        df4 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "rmse"}

        df5 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "mse"}

        df6 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "mape_"}

        df7 = {'rotate_angle': None,
               'box_detect_confidence_top_cam': None,
               'box_detect_confidence_side_cam': None,
               'arrow_position_encode': None,
               'arrow_direction_encode': None,
               'angle_type_encode': None,
               'parameter': "accuracy"}

        for parameter_type, value in df2.items():
            if parameter_type == "rotate_angle":
                true_value = list(data_frame_gt["angle_gt_" + str(box_number + 1)])
                pred_value = list(data_frame_gt["rotate_angle"])

                if len(true_value) == 0 or len(pred_value) == 0:
                    mse = None
                    rmse = None
                    mape_ = None
                else:

                    mse, rmse, mape_ = calculate_mse_rmse_r2_score(true_value, pred_value)

                df4[parameter_type] = rmse
                df5[parameter_type] = mse
                df6[parameter_type] = mape_


            elif parameter_type == "box_detect_confidence_top_cam":
                true_value = [100] * len(data_frame_gt)
                pred_value = data_frame_gt["box_detect_confidence_top_cam"]
                if len(true_value) == 0 or len(pred_value) == 0:
                    mse = None
                    rmse = None
                    mape_ = None
                else:

                    mse, rmse, mape_ = calculate_mse_rmse_r2_score(true_value, pred_value)
                df4[parameter_type] = rmse
                df5[parameter_type] = mse
                df6[parameter_type] = mape_

            elif parameter_type == "box_detect_confidence_side_cam":
                true_value = [100] * len(data_frame_gt)
                pred_value = data_frame_gt["box_detect_confidence_side_cam"]
                if len(true_value) == 0 or len(pred_value) == 0:
                    mse = None
                    rmse = None
                    mape_ = None
                else:

                    mse, rmse, mape_ = calculate_mse_rmse_r2_score(true_value, pred_value)
                df4[parameter_type] = rmse
                df5[parameter_type] = mse
                df6[parameter_type] = mape_

            elif parameter_type == "arrow_position_encode":
                true_value = data_frame_gt["arrow_position_gt_" + str(box_number + 1)]
                pred_value = data_frame_gt["arrow_position_encode"]
                precission, recall , accuracy= calculate_pre_recall_accuracy(true_value, pred_value)

                df2[parameter_type] = precission
                df3[parameter_type] = recall
                df7[parameter_type] = accuracy

            elif parameter_type == "arrow_direction_encode":
                true_value = data_frame_gt["arrow_direction_gt_" + str(box_number + 1)]
                pred_value = data_frame_gt["arrow_direction_encode"]
                precission, recall, accuracy = calculate_pre_recall_accuracy(true_value, pred_value)

                df2[parameter_type] = precission
                df3[parameter_type] = recall
                df7[parameter_type] = accuracy

            elif parameter_type == "angle_type_encode":
                try:
                    true_value = data_frame_gt["angle_type_gt_" + str(box_number + 1)]
                except:
                    true_value = data_frame_gt["angle_type_type_gt_" + str(box_number + 1)]

                pred_value = data_frame_gt["angle_type_encode"]
                precission, recall, accuracy = calculate_pre_recall_accuracy(true_value, pred_value)

                df2[parameter_type] = precission
                df3[parameter_type] = recall
                df7[parameter_type] = accuracy

        analysis_data = analysis_data.append(df2, ignore_index=True)
        analysis_data = analysis_data.append(df3, ignore_index=True)
        analysis_data = analysis_data.append(df4, ignore_index=True)
        analysis_data = analysis_data.append(df5, ignore_index=True)
        analysis_data = analysis_data.append(df6, ignore_index=True)
        analysis_data = analysis_data.append(df7, ignore_index=True)
        print(csv_name)

        if save_result:
            analysis_data.to_csv(
                r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/statistical_data_field_test/' + csv_name,
                index=True, header=True)


if __name__ == '__main__':
    calculate_statical_analysis_value(
        "/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/processed_data_field_test/preprocessed_0033.csv",
        "/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/processed_data_field_test/preprocessed_gt_0033.csv")

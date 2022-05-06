import tkinter as tk
from tkinter import ttk


class GuiObject:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Orientation Result')
        self.window.geometry('900x600')
        self.font = "Arial"

        self.canvas_for_heading = tk.Canvas(self.window, width=900, height=100, bg="#999999")
        self.canvas_for_heading.pack()

        self.canvas_for_raw_data = tk.Canvas(self.window, width=900, height=50, bg="#555555")
        self.canvas_for_raw_data.pack()

        self.tree = ttk.Treeview(self.window)  # creat Table

        self.tree["columns"] = ("Box Number", "Angle", "Arrow Position", "Direction")
        self.tree.column('#0', width=0, stretch=tk.NO)
        self.tree.column("Box Number", width=225, stretch=tk.YES)  # define
        self.tree.column("Angle", width=225, stretch=tk.YES)  # define
        self.tree.column("Direction", width=225, stretch=tk.YES)
        self.tree.column("Arrow Position", width=225, stretch=tk.YES)
        vsb = ttk.Scrollbar(self.window, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=vsb.set, selectmode="extended")

        self.tree.heading("Box Number", text="Box Number")  # display headings
        self.tree.heading("Angle", text="Angle")  # display headings
        self.tree.heading("Direction", text="Direction")
        self.tree.heading("Arrow Position", text="Arrow Position")
        self.tree.pack(fill="both", expand=tk.YES)

        self.canvas_for_final_decision = tk.Canvas(self.window, width=900, height=300, bg="#333333")
        self.canvas_for_final_decision.pack()

        self.headings = self.canvas_for_heading.create_text(450, 25, font=(self.font, 20, 'bold'),
                                                            text='Box Orientation')
        self.headings_describe = self.canvas_for_heading.create_text(450, 60, font=(self.font, 12),
                                                                     text='This result is for container unloading system ')

        self.headings_raw_data = self.canvas_for_raw_data.create_text(450, 25, font=(self.font, 20, 'bold'),
                                                                      text='Raw data')

        self.headings_final_result = self.canvas_for_final_decision.create_text(450, 35, font=(self.font, 20, 'bold'),
                                                                                text='Final Result', fill="white")

        self.content_final_result_box_number = self.canvas_for_final_decision.create_text(450, 65, font=(self.font, 12),
                                                                                          text='Box Number :',
                                                                                          fill="white")
        self.content_final_result_ori_type = self.canvas_for_final_decision.create_text(450, 90, font=(self.font, 12),
                                                                                        text='Orientation Type :',
                                                                                        fill="white")
        self.content_final_result_angle_rotation = self.canvas_for_final_decision.create_text(450, 115,
                                                                                              font=(self.font, 12),
                                                                                              text='Angle of Rotation :',
                                                                                              fill="white")
        self.content_final_result_arrow_pos = self.canvas_for_final_decision.create_text(450, 155,
                                                                                         font=(self.font, 12),
                                                                                         text='Position of Angle  :',
                                                                                         fill="white")
        self.content_final_result_arrow_dir = self.canvas_for_final_decision.create_text(450, 195,
                                                                                         font=(self.font, 12),
                                                                                         text='Direction of Angle  :',
                                                                                         fill="white")
        self.content_final_result_box_number_value = self.canvas_for_final_decision.create_text(650, 60,
                                                                                                font=(self.font, 12),
                                                                                                text='Box number  :',
                                                                                                fill="white")
        self.content_final_result_ori_type_value = self.canvas_for_final_decision.create_text(650, 90,
                                                                                              font=(self.font, 12),
                                                                                              text='Orientation Type :',
                                                                                              fill="white")
        self.content_final_result_angle_rotation_value = self.canvas_for_final_decision.create_text(650, 115,
                                                                                                    font=(
                                                                                                        self.font, 12),
                                                                                                    text='Angle of Rotation :',
                                                                                                    fill="white")
        self.content_final_result_arrow_pos_value = self.canvas_for_final_decision.create_text(650, 155,
                                                                                               font=(self.font, 12),
                                                                                               text='Position of Angle  :',
                                                                                               fill="white")
        self.content_final_result_arrow_dir_value = self.canvas_for_final_decision.create_text(650, 195,
                                                                                               font=(self.font, 12),
                                                                                               text='Direction of Angle  :',
                                                                                               fill="white")

        self.update_final_result()
        # self.window.mainloop()

    def update_final_result(self, box_umber="null", angle=0, arrow_dir="null", arrow_position="null", ori_type="null"):

        try:

            self.canvas_for_final_decision.itemconfig(self.content_final_result_box_number_value, text=str(box_umber))
            self.canvas_for_final_decision.itemconfig(self.content_final_result_arrow_dir_value, text=str(arrow_dir))
            self.canvas_for_final_decision.itemconfig(self.content_final_result_arrow_pos_value,
                                                      text=str(arrow_position))
            self.canvas_for_final_decision.itemconfig(self.content_final_result_angle_rotation_value, text=str(angle))
            self.canvas_for_final_decision.itemconfig(self.content_final_result_ori_type_value, text=str(ori_type))

        except:
            print("update_raw_data in except")

    def update_table(self, box_umber="null", angle=0, arrow_dir="null", arrow_position="null"):
        try:
            self.tree.insert("", 'end', text="person" + str(1 + 1), values=(
                str(box_umber), str(angle), str(arrow_dir), str(arrow_position)))
        except:
            pass

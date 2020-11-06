from tracking import full_vid_main

no_cams = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]

for no_cam in no_cams:
    full_vid_main(no_cam=str(no_cam), params_path="params" \
                              , save_path = "process" \
                              , json_path = "/data/test_data" \
                              , full_vid_path="/data/test_data")

with open('/data/submission_output/submission.txt', 'w') as f:
    for no_cam in no_cams:
        with open('process/total_' + no_cam + '.txt') as no_cam_f:
                f.write(no_cam_f.read())

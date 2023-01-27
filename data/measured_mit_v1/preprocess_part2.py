from scipy import io
import numpy as np
import pdb
import scipy.signal


import matplotlib.pyplot as plt


if __name__ == '__main__':
    pdb.set_trace()
    raw = io.loadmat(f"raw/ultrasound_data_with_estimatedPP.mat")['ultrasound_data']

    name_all = []
    v_all = []
    a_all = []

    pwv_all = []
    age_all = []
    comp_all = []
    z0_all = []
    pp_all = []
    bp_shape_all = []

    map_all = []

    resample_len_per_beat = 100
    id_all = []
    dbp_all = []
    heartrate_all = []
    area_all = []

    gender_all = []
    height_all = []
    weight_all = []

    diameter_complete_all = []
    diameter_complete_time_frame_all = []
    velocity_complete_all = []
    velocity_complete_time_frame_all = []

    for subject_id in range(23):
        if subject_id == 20 or subject_id == 21 or subject_id == 22 or subject_id == 23:
            continue
        print(subject_id)
        data = raw[subject_id][0][10][0][0]
        id_all.append(subject_id)

        velocity = data[1].flatten()
        velocity = velocity[np.logical_not(np.isnan(velocity))]

        diameter = data[3].flatten()
        diameter = diameter[np.logical_not(np.isnan(diameter))]

        diameter = scipy.signal.resample(diameter, resample_len_per_beat)
        velocity = scipy.signal.resample(velocity, resample_len_per_beat)

        diameter_max_index = np.argmax(diameter)
        diameter = np.concatenate(
            (diameter[diameter_max_index:], diameter[:diameter_max_index]))

        velocity_max_index = np.argmax(velocity)
        velocity = np.concatenate(
            (velocity[velocity_max_index:], velocity[:velocity_max_index]))

        name = raw[subject_id][0][0][0]

        diameter_complete = raw[subject_id][0][1][0][0][0]
        diameter_complete_all.append(diameter_complete)

        diameter_complete_time_frame = raw[subject_id][0][1][0][0][2]
        diameter_complete_time_frame_all.append(diameter_complete_time_frame)

        velocity_complete = raw[subject_id][0][2][0][0][0]
        velocity_complete_all.append(velocity_complete)

        velocity_complete_time_frame = raw[subject_id][0][2][0][0][2]
        velocity_complete_time_frame_all.append(diameter_complete_time_frame)

        print(name)
        data_supine_bp = raw[subject_id][0][7][0]
        SBP = data_supine_bp[0][0][0][0][0]
        SBP = SBP[np.logical_not(np.isnan(SBP))]
        SBP_avg = SBP.mean()

        DBP = data_supine_bp[0][0][0][0][1]
        DBP = DBP[np.logical_not(np.isnan(DBP))]
        DBP_avg = DBP.mean()

        ppressure = SBP_avg - DBP_avg

        area = np.pi * np.square(diameter) / 4
        compliance = (max(area) - min(area)) / ppressure
        bp_shape = (area - area.mean()) / compliance

        v_all.append(velocity)
        bp_shape_all.append(bp_shape)
        name_all.append(name)

        map = DBP_avg + (area.mean() - min(area)) / (max(area) - min(area)) * (
                    SBP_avg - DBP_avg)
        map_all.append(map)
        dbp_all.append(DBP_avg)
        area_all.append(area)

        anthro = raw[subject_id][0][5][0][0]
        age = anthro[0][0][0]
        if age == 0:
            age_all.append(30)
        else:
            age_all.append(age)

        if anthro[1][0][0][0] == 'male':
            gender_all.append(0)
        else:
            gender_all.append(1)

        height_all.append(anthro[2][0][0])
        weight_all.append(anthro[3][0][0])

    np.save(f'./npy/measured_mit_v1_part2_shape_all.npy', np.array(bp_shape_all))
    np.save(f'./npy/measured_mit_v1_part2_map_all.npy', np.array(map_all))
    np.save(f'./npy/measured_mit_v1_part2_v_all.npy', np.array(v_all))
    np.save(f'./npy/measured_mit_v1_part2_name_all.npy', np.array(name_all))
    np.save(f'./npy/measured_mit_v1_part2_id_all.npy', np.array(id_all))
    np.save(f'./npy/measured_mit_v1_part2_dbp_all.npy', np.array(dbp_all))
    np.save(f'./npy/measured_mit_v1_part2_area_all.npy', np.array(area_all))

    np.save(f'./npy/measured_mit_v1_part2_age_all.npy', np.array(age_all))
    np.save(f'./npy/measured_mit_v1_part2_gender_all.npy', np.array(gender_all))
    np.save(f'./npy/measured_mit_v1_part2_height_all.npy', np.array(height_all))
    np.save(f'./npy/measured_mit_v1_part2_weight_all.npy', np.array(weight_all))

    np.save(f'./npy/measured_mit_v1_part2_diameter_complete_all.npy', np.array(diameter_complete_all))
    np.save(f'./npy/measured_mit_v1_part2_diameter_complete_time_frame_all.npy', np.array(diameter_complete_time_frame_all))
    np.save(f'./npy/measured_mit_v1_part2_velocity_complete_all.npy', np.array(velocity_complete_all))
    np.save(f'./npy/measured_mit_v1_part2_velocity_complete_time_frame_all.npy', np.array(velocity_complete_time_frame_all))

    np.save(f'./npy/measured_mit_v1_part2_heartrate_all.npy', np.array(heartrate_all))

    print('finished')

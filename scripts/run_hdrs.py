import pandas as pd
from project_catalog.utils import get_galactic_binary_names
from project_catalog.galactic_binary import GalacticBinary
from project_catalog.highest_density_region import compute_1d_hdrs, compute_8d_hdr_all_injections
from project_catalog.highest_density_region import compute_6d_hdr_all_injections
import fire
from tqdm import tqdm

def get_hdrs(gb_index: int, plot_dir: str|None = './plots/') -> pd.DataFrame:
    print(f'Computing HDRs for GB index {gb_index}')
    names = get_galactic_binary_names()
    hdr_data = {'Name': [], 'Candidate': [], 'SNR': [], '8D HDR': [],
                '6D HDR': [], 'Frequency HDR': [],
                'Amplitude HDR': [], 'Inclination HDR': [],
                'Initial Phase HDR': [], 'Ecliptic Latitude HDR': [],
                'Ecliptic Longitude HDR': [], 'Polarization HDR': [],
                'Frequency Derivative HDR': []}
    gb = GalacticBinary.load_feather(names[gb_index])
    for i in range(gb.injections.shape[0]):
        hdrs = compute_1d_hdrs(gb, i)
        hdr_data['Frequency HDR'].append(hdrs['Frequency'])
        hdr_data['Amplitude HDR'].append(hdrs['Amplitude'])
        hdr_data['Inclination HDR'].append(hdrs['Inclination'])
        hdr_data['Initial Phase HDR'].append(hdrs['Initial Phase'])
        hdr_data['Ecliptic Latitude HDR'].append(hdrs['Ecliptic Latitude'])
        hdr_data['Ecliptic Longitude HDR'].append(hdrs['Ecliptic Longitude'])
        hdr_data['Polarization HDR'].append(hdrs['Polarization'])
        hdr_data['Frequency Derivative HDR'].append(hdrs['Frequency Derivative'])
        hdr_data['Name'].append(gb.name)
        hdr_data['Candidate'].append(gb.injections.iloc[i]['Name'])
        hdr_data['SNR'].append(gb.injections.iloc[i]['SNR'])
    hdr_8d = compute_8d_hdr_all_injections(gb, plot_dir=plot_dir)
    for i in range(gb.injections.shape[0]):
        hdr_data['8D HDR'].append(hdr_8d[i])
    hdr_6d = compute_6d_hdr_all_injections(gb, plot_dir=plot_dir)
    for i in range(gb.injections.shape[0]):
        hdr_data['6D HDR'].append(hdr_6d[i])
    hdrs = pd.DataFrame(hdr_data)
    print(f'Saving HDRs to ../results/hdrs_{gb.name}.feather')
    hdrs.to_feather(f'../results/hdrs_{gb.name}.feather')


if __name__ == "__main__":
    uncomputed_indices = [410, 670, 747, 1010, 1065, 1079, 1145, 1198, 2061, 2065, 2092, 2176, 2199, 2254, 2281, 2353, 2356, 2364, 2376, 2488, 2536, 2545, 2622, 2729, 3039, 3274, 3371, 3374, 3620, 3621, 3672, 3673, 3685, 3862, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 5188, 5278, 5366, 5601, 5648, 5654, 5870, 6176, 6557, 6582, 6695, 6700, 6705, 6709, 6974, 7182, 7452, 7455, 7555, 7656, 7829, 7832, 7833, 7834, 7835, 7836, 7837, 7838, 7839, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883, 7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7896, 7897, 7898, 7899, 7900, 8083, 8096, 8114, 8127, 8137, 8143, 8144, 8145, 8146, 8344, 8400, 8401, 8402, 8403, 8404, 8405, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414, 8419, 8420, 8427, 8497, 8533, 8549, 8571, 8572, 8573, 8574, 8575, 8576, 8579, 8586, 8683, 8715, 8725, 8740, 8746, 8747, 8748, 8749, 8750, 8751, 8752, 8753, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8977, 8982, 8983, 8984, 8985, 8986, 8987, 8988, 8989, 8990, 8991, 8992, 8993, 9078, 9079, 9080, 9156, 9161, 9162, 9166, 9167, 9168, 9169, 9170, 9171, 9172, 9173, 9174, 9242, 9253, 9301, 9302, 9328, 9329, 9330, 9377, 9390, 9391, 9392, 9393, 9454, 9455, 9456, 9457, 9458, 9459, 9460, 9519, 9526, 9527, 9571, 9572, 9581, 9582, 9583, 9584, 9616, 9623, 9624, 9625, 9665, 9666, 9667, 9668, 9669, 9670, 9671, 9672, 9673, 9676, 9725, 9726, 9727, 9728, 9774, 9775, 9776, 9811, 9848, 9849, 9850, 9851, 9852, 9853, 9854, 9889, 9909, 9916, 9917, 9918, 9928, 9953, 9954, 9995, 9996, 10019, 10020, 10021, 10050, 10051, 10052, 10087, 10088, 10109, 10114, 10117, 10118, 10119, 10120, 10133, 10154, 10155, 10184, 10185, 10212, 10307, 10308, 10329, 10330, 10331, 10355, 10356, 10357, 10401, 10419, 10436, 10441, 10442, 10479, 10496, 10499, 10500, 10513, 10514, 10533, 10534, 10550, 10573, 10574, 10575, 10576, 10577, 10578, 10593, 10612, 10623, 10645, 10686, 10687, 10688, 10695, 10702, 10704, 10708, 10727, 10729, 10744, 10747, 10763, 10789, 10791, 10817, 10818, 10834, 10853, 10855, 10952, 10954, 10969, 11113, 11158, 11159, 11205, 11215, 11223, 11263, 11278, 11309, 11337, 11352, 11390, 11393, 11405, 11409, 11426, 11434, 11476, 11477, 11531, 11538, 11595, 11596, 11597, 11598, 11599, 11600]
    for index in tqdm(uncomputed_indices):
        get_hdrs(gb_index=index, plot_dir='./plots/')
    # fire.Fire(get_hdrs)
    # Example usage: python scripts/run_hdrs.py --gb_index=0 --plot_dir='./plots/'

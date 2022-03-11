import numpy as np
import math
import nibabel as nib
from scipy import interpolate


def DiODe(imgFp, marker, segment):
    # Input arguments:
    # (1) imgFp: string. the file name of the input Post-CT image
    # (2) marker: list. The position of marker. Example: [243, 297, 107]
    # (3) segment: list. The position of the proximal segments. Example: [248, 299, 100]

    print("** Start **")
    marker = np.asarray(marker)
    segment = np.asarray(segment)
    traj = marker - segment
    traj = traj / np.linalg.norm(traj)
    polar_angle = math.degrees(math.acos(np.dot(traj, [0, 0, 1])))
    assert abs(polar_angle) < 50, f"The angle between the lead and the slice " \
                                  f"normal is {polar_angle} degrees.\nNote that angles " \
                                  f"> 50 degrees could cause inaccurate orientation estimation."
    img = nib.load(imgFp)
    V = img.get_fdata()
    header_info = img.header
    dims = header_info['dim'][1:4]
    spacings = header_info['pixdim'][1:4]
    assert spacings[0] == spacings[1], "The X and Y axis should have the same spacings"

    radius1 = 3  # radius for analysis at marker (default 3 mm)
    radius2 = 8  # radius for analysis at proximal segments (default 8 mm)

    # Initial orientation estimation at marker position
    slice1 = V[:, :, marker[2] - 1]  # slice at the marker
    print("** Analyzing circular intensity profile at marker **")
    n = 360
    x = np.zeros(n)
    y = np.zeros(n)
    val = np.zeros(n)
    f = interpolate.interp2d(np.arange(dims[0]), np.arange(dims[1]), slice1, kind='linear')
    for theta in range(360):
        x[theta] = radius1 / spacings[0] * math.sin(math.radians(theta)) + marker[1]
        y[theta] = radius1 / spacings[0] * math.cos(math.radians(theta)) + marker[0]
        val[theta] = f(x[theta], y[theta])

    M = np.fft.fft(val)
    m = max(abs(M))
    M[np.where(abs(M) < 0.99 * m)] = 0
    denoised_val = np.fft.ifft(M)
    angles = denoised_val.argsort()[-2:]
    print(f"Peak angles (in degrees): {angles[0]}, {angles[1]}")

    phase = 0
    for i in range(359):
        if denoised_val[i] == 0:
            phase = i
            break
        elif denoised_val[i] * denoised_val[i + 1] < 0:
            phase = i + 0.5
            break
    dangles = np.array([int(round(angles[0] + (45 - phase / 2))), int(round(angles[1] + (45 - phase / 2)))])
    print(f"** Denoised peak angles (in degrees): {dangles[0]}, {dangles[1]}")

    print("** Analyzing circular intensity profile at proximal segments **")
    slice2 = V[:, :, segment[2] - 1]
    x = np.zeros(n)
    y = np.zeros(n)
    for theta in range(360):
        x[theta] = radius2 / spacings[0] * math.sin(math.radians(theta)) + segment[1]
        y[theta] = radius2 / spacings[0] * math.cos(math.radians(theta)) + segment[0]

    corrections = np.arange(-30, 31)
    n = corrections.shape[0]
    val = np.zeros(n)
    init_angle = angles[0] + 90
    init_angles = np.linspace(init_angle + corrections[0], init_angle + corrections[-1], n)
    intervals = [60, 120, 180, 240, 300, 360]
    f2 = interpolate.interp2d(np.arange(dims[0]), np.arange(dims[1]), slice2, kind='linear')
    for i in range(n):
        marker_angle = init_angles[i]
        for j in range(6):
            x = radius2 * math.sin(math.radians(marker_angle + intervals[j])) + segment[1]
            y = radius2 * math.cos(math.radians(marker_angle + intervals[j])) + segment[0]
            val[i] += f2(x, y)

    val /= 6
    final_correction = corrections[np.argmin(val)]
    observed_angles = angles + final_correction - 90
    gamma = math.radians(observed_angles[0])

    print(f"Corrected peak angle (in degrees): {final_correction}")
    print(f"Observed roll angles at the axial plane (in degrees): {observed_angles[0]}, {observed_angles[1]}")

    beta = math.asin(traj[0])  # yaw
    alpha = math.asin(traj[1] / math.cos(beta))  # pitch
    gamma = math.atan((math.sin(gamma) * math.cos(alpha)) / \
                      (math.cos(gamma) * math.cos(beta) - math.sin(alpha) * math.sin(beta) * math.sin(gamma)))
    print(f"Pitch, yaw, roll angles (in degrees): {math.degrees(alpha)}, {math.degrees(beta)}, {math.degrees(gamma)}")

    marker_orientation = [-math.sin(gamma) * math.cos(beta),
                          math.cos(gamma) * math.cos(alpha) + math.sin(alpha) * math.sin(beta) * math.sin(gamma),
                          -math.cos(gamma) * math.sin(alpha) + math.sin(gamma) * math.sin(beta) * math.cos(alpha)]
    print("** Estimated marker orientation **")
    print(f"Isotropic: [{marker_orientation[0]}, {marker_orientation[1]}, {marker_orientation[2]}]")

    marker_orientation /= spacings
    marker_orientation = marker_orientation / np.linalg.norm(marker_orientation)
    print(f"With current spacing: [{marker_orientation[0]}, {marker_orientation[1]}, {marker_orientation[2]}]")
    print("** Done **")


if __name__ == "__main__":
    imgFp = "/home/liuh26/Documents/TestData/test.nii"
    marker = [243, 297, 107]
    proxSg = [248, 299, 100]
    DiODe(imgFp, marker, proxSg)

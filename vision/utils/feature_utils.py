from scipy.spatial import distance


def eye_aspect_ratio(contour):
    """
    The weight of EAR is calculated by six point contour.

    Args:
        A list of six points.
    Returns:
        A weight of EAR.
    """
    A = distance.euclidean(contour[1], contour[5])
    B = distance.euclidean(contour[2], contour[4])
    C = distance.euclidean(contour[0], contour[3])
    EAR = (A + B) / (2.0 * C)
    return EAR




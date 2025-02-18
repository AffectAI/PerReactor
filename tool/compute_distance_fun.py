from tslearn.metrics import dtw
def compute_distance(a, b):
	res = 0
	for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
		res += weight * dtw(a[ : , st : ed], b[ : , st : ed])
	return res

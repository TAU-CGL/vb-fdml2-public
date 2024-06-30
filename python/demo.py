import math

import vbfdml2 as vbfdml

n = 200
be = vbfdml.BoxExtent(2.0, 2.0, 2.0 * math.pi, 0.0, 0.0, 0.0)

scene = [
	vbfdml.Segment(-.66, -.66, .0, .33),
    vbfdml.Segment(.0, .33, 1., .66),
    vbfdml.Segment(1., .66, .66, .0),
    vbfdml.Segment(.66, .0, 1., -.66),
    vbfdml.Segment(1., -.66, -.66, -.66),
]

preprocess, mask = vbfdml.preprocess_scene(n, be, scene, True)
m1 = vbfdml.marching_voxels(preprocess, be, 0, 0, 0, 0.5, True)
# m1 = vbfdml.conv3d(m1, True)

m2 = vbfdml.marching_voxels(preprocess, be, 0, 0, math.pi / 4, 0.5, True)
# m2 = vbfdml.conv3d(m2, True)

m3 = vbfdml.marching_voxels(preprocess, be, 0, 0, math.pi / 2, 0.5, True)
# m3 = vbfdml.conv3d(m3, True)

m4 = vbfdml.marching_voxels(preprocess, be, 0, 0, math.pi, 1.0, True)
# m4 = vbfdml.conv3d(m4, True)

isect = m1
isect = vbfdml.do_intersect(isect, m2, True)
isect = vbfdml.do_intersect(isect, m3, True)
isect = vbfdml.do_intersect(isect, m4, True)
# isect = vbfdml.do_intersect(isect, mask, True)

isect.ExportOBJ("./tmp/test.obj", be)

predictions = vbfdml.predict(isect, be)
for pred in predictions:
    print(pred)
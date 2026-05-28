import numpy as np

import camera as camera_module
import params


def make_camera():
    return camera_module.camera(
        fx=481.20,
        fy=480.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )


def test_intrinsics_and_image_size_scale_consistently_across_pyramid():
    cam = make_camera()

    for lvl in range(params.MAX_LEVELS):
        scale = 2**lvl
        assert cam.width[lvl] == params.IMAGE_WIDTH // scale
        assert cam.height[lvl] == params.IMAGE_HEIGHT // scale
        assert cam.fx[lvl] == cam.fx[0] / scale
        assert cam.fy[lvl] == cam.fy[0] / scale
        assert cam.cx[lvl] == cam.cx[0] / scale
        assert cam.cy[lvl] == cam.cy[0] / scale


def test_backproject_project_roundtrip_is_pixel_accurate_at_each_level():
    cam = make_camera()

    for lvl in range(params.MAX_LEVELS):
        pixel = np.array(
            [
                min(cam.width[lvl] - 2.0, cam.cx[lvl] + 7.25),
                min(cam.height[lvl] - 2.0, cam.cy[lvl] - 5.5),
            ]
        )
        inv_depth = 0.4

        ray = np.array(
            [
                cam.fxinv[lvl] * pixel[0] + cam.cxinv[lvl],
                cam.fyinv[lvl] * pixel[1] + cam.cyinv[lvl],
                1.0,
            ]
        )
        point = ray / inv_depth
        projected = np.array(
            [
                cam.fx[lvl] * point[0] / point[2] + cam.cx[lvl],
                cam.fy[lvl] * point[1] / point[2] + cam.cy[lvl],
            ]
        )

        np.testing.assert_allclose(projected, pixel, atol=1e-10)

import cv2
import numpy as np
import pytest

import common
import frameData
import params


def test_subpixel_value_uses_bilinear_interpolation_for_scalar_images():
    image = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)

    value = common.getSubPixelValue(image, np.array([0.25, 0.5]))

    assert value == pytest.approx(12.5)


def test_subpixel_value_interpolates_vector_fields_componentwise():
    field = np.array(
        [
            [[0.0, 0.0], [10.0, 100.0]],
            [[20.0, 200.0], [30.0, 300.0]],
        ],
        dtype=np.float32,
    )

    value = common.getSubPixelValue(field, np.array([0.25, 0.5]))

    np.testing.assert_allclose(value, [12.5, 125.0])


def test_frame_pyramid_has_expected_shapes_and_constant_images_have_zero_gradient():
    image = np.full((480, 640), 37, dtype=np.uint8)
    frame = frameData.frameData()

    frame.setImage(image)

    for lvl in range(params.MAX_LEVELS):
        expected_shape = (params.IMAGE_HEIGHT // (2**lvl), params.IMAGE_WIDTH // (2**lvl))
        assert frame.image[lvl].shape == expected_shape
        assert frame.imageDerivative[lvl].shape == expected_shape + (2,)
        np.testing.assert_allclose(frame.imageDerivative[lvl], 0.0)


def test_frame_pyramid_uses_area_resampling_when_building_vo_images():
    image = np.indices((480, 640)).sum(axis=0).astype(np.uint8)
    frame = frameData.frameData()

    frame.setImage(image)

    expected = cv2.resize(
        image,
        (params.IMAGE_WIDTH, params.IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    np.testing.assert_array_equal(frame.image[0], expected)

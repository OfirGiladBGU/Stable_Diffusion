import numpy as np
from typing import Dict, Union


def project_3d_to_2d(data_3d: np.ndarray,
                     projection_options: dict[str, bool],
                     source_data_filepath=None,
                     component_3d: np.ndarray = None) -> Union[Dict, Dict[str, np.ndarray]]:
    projections = dict()

    rotated_data_3d = data_3d
    rotated_component_3d = component_3d

    if source_data_filepath is None:
        pass # No need for rotation

    # Medical data (nii.gz) has different axis order
    elif str(source_data_filepath).endswith(".nii.gz") is True:
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 2))  # For OpenCV compatibility
        if component_3d is not None:
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 2))

    # Other data formats
    else:
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 2))  # For OpenCV compatibility
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 1))
        if component_3d is not None:
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 2))
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 1))

    # Front projection (XZ plane)
    if projection_options.get("front", False) is True:
        flipped_data_3d = rotated_data_3d

        # Option 1
        # projections["front_image"] = np.max(data_3d, axis=2)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
           pass  # No need for rotation

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["front_image"] = depth_projections.get("image", None)
        projections["front_components"] = depth_projections.get("components", None)  # Optional

    # Back projection (XZ plane)
    if projection_options.get("back", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["back_image"] = np.max(flipped_data_3d, axis=2)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=2, axes=(1, 2))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["back_image"] = depth_projections.get("image", None)
        projections["back_components"] = depth_projections.get("components", None)  # Optional


    # Top projection (XY plane)
    if projection_options.get("top", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(0, 1))

        # Option 1
        # projections["top_image"] = np.max(data_3d, axis=1)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(0, 1))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["top_image"] = depth_projections.get("image", None)
        projections["top_components"] = depth_projections.get("components", None)  # Optional


    # Bottom projection (XY plane)
    if projection_options.get("bottom", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 0))

        # Option 1
        # projections["bottom_image"] = np.max(flipped_data_3d, axis=1)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 0))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["bottom_image"] = depth_projections.get("image", None)
        projections["bottom_components"] = depth_projections.get("components", None)  # Optional

    # Right projection (YZ plane)
    if projection_options.get("right", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))

        # Option 1
        # projections["right_image"] = np.max(flipped_data_3d, axis=0)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 2))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["right_image"] = depth_projections.get("image", None)
        projections["right_components"] = depth_projections.get("components", None)  # Optional

    # Left projection (YZ plane)
    if projection_options.get("left", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(2, 1))

        # Option 1
        # projections["left_image"] = np.max(data_3d, axis=0)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(2, 1))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["left_image"] = depth_projections.get("image", None)
        projections["left_components"] = depth_projections.get("components", None)  # Optional

    return projections


########################
# 3D to 2D projections #
########################
def _calculate_depth_projection(data_3d: np.ndarray, component_3d: np.ndarray = None, axis: int = 0):
    depth_projection = np.argmax(data_3d, axis=axis)
    max_projection = np.max(data_3d, axis=axis)
    max_axis_index = data_3d.shape[axis] - 1

    grayscale_depth_projection = np.where(
        max_projection > 0,
        np.round(255 * (1 - (depth_projection / max_axis_index))),
        0
    ).astype(np.uint8)

    depth_projects = dict()
    depth_projects["image"] = grayscale_depth_projection
    if component_3d is not None:
        components_depth_projection = np.zeros_like(grayscale_depth_projection)
        for i in range(grayscale_depth_projection.shape[0]):
            for j in range(grayscale_depth_projection.shape[1]):
                if grayscale_depth_projection[i, j] > 0:
                    if axis == 0:
                        components_depth_projection[i, j] = component_3d[depth_projection[i, j], i, j]
                    elif axis == 1:
                        components_depth_projection[i, j] = component_3d[i, depth_projection[i, j], j]
                    elif axis == 2:
                        components_depth_projection[i, j] = component_3d[i, j, depth_projection[i, j]]
                    else:
                        raise ValueError("Invalid axis")
        depth_projects["components"] = components_depth_projection

    return depth_projects

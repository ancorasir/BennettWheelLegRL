import torch

# from omni.isaac.core.utils.torch.transformations import tf_matrices_from_poses
from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul, quat_axis
from scipy.spatial.transform import Rotation

def rot_matrices_from_quat(orientations: torch.Tensor, device=None) -> torch.Tensor:
    """
        Args:
            orientations: quaternions of shape (N, 4)
        
        Returns:
            corresponding rotation matrices of shape (N, 3, 3)
    """
    result_mat = torch.zeros([orientations.shape[0], 3, 3], dtype=torch.float32, device=device)
    r = Rotation.from_quat(orientations[:, [1, 2, 3, 0]].detach().cpu().numpy())
    result_mat[:, :3, :3] = torch.from_numpy(r.as_matrix()).float().to(device)

    return result_mat

def quat_from_rot_matrices(rot_matrices: torch.Tensor, device=None) -> torch.Tensor:
    """
        Args:
            rot_matrices: rotation matrices of shape (N, 3, 3)

        Returns:
            tensors of quaternions of shape (N, 4)
    """
    r = Rotation.from_matrix(rot_matrices.detach().cpu().numpy())
    quats = r.as_quat()[:, [3, 0, 1, 2]]
    return torch.from_numpy(quats).float().to(device)

def rotate_orientations(rotations: torch.Tensor, orientations: torch.Tensor, device=None) -> torch.Tensor:
    """
        Args:
            rotations: the rotations to be applied to the orientations; need to be quaternions
                of shape (N, 4)
            orientations: the quaternions to be rotated of shape (N, 4)

        Returns:
            The rotated quaternions of shape (N, 4)
    """
    # Transform the quaternion to rotation matrices
    rot_matrices = rot_matrices_from_quat(rotations, device) # (N, 3, 3)
    orient_rot_matrices = rot_matrices_from_quat(orientations, device) # (N, 3, 3)

    # Apply rotations to the orientations
    rotated_orient_rot_matrices = torch.bmm(rot_matrices, orient_rot_matrices)

    result_quat = quat_from_rot_matrices(rotated_orient_rot_matrices, device)
    # Make the scalar part to be positive
    positive_real_part_quat = torch.where((result_quat[:, 0] < 0.0).view(-1, 1),
                                          -result_quat,
                                          result_quat)

    # Transform back to quaternions
    return positive_real_part_quat

def inverse_rotate_orientations(rotations: torch.Tensor, orientations: torch.Tensor, device=None)-> torch.Tensor:
    """
        Args:
            rotations: the rotations to be applied to the orientations; need to be quaternions
                of shape (N, 4)
            orientations: the quaternions to be rotated of shape (N, 4)

        Returns:
            The inversely rotated quaternions of shape (N, 4)
    """
    # Transform the quaternion to rotation matrices
    inverse_rot_matrices = torch.transpose(rot_matrices_from_quat(rotations, device), dim0=1, dim1=2) # (N, 3, 3)
    orient_rot_matrices = rot_matrices_from_quat(orientations, device) # (N, 3, 3)

    # Apply rotations to the orientations
    rotated_orient_rot_matrices = torch.bmm(inverse_rot_matrices, orient_rot_matrices)

    result_quat = quat_from_rot_matrices(rotated_orient_rot_matrices, device)
    # Make the scalar part to be positive
    positive_real_part_quat = torch.where((result_quat[:, 0] < 0.0).view(-1, 1),
                                          -result_quat,
                                          result_quat)

    # Transform back to quaternions
    return positive_real_part_quat

def tf_matrices_from_poses(translations: torch.Tensor, orientations: torch.Tensor, device=None) -> torch.Tensor:
    """[summary]

    Args:
        translations (Union[np.ndarray, torch.Tensor]): translations with shape (N, 3).
        orientations (Union[np.ndarray, torch.Tensor]): quaternion representation (scalar first) with shape (N, 4).

    Returns:
        Union[np.ndarray, torch.Tensor]: transformation matrices with shape (N, 4, 4)
    """
    result = torch.zeros([orientations.shape[0], 4, 4], dtype=torch.float32, device=device)
    rot_mat = rot_matrices_from_quat(orientations, device)
    result[:, :3, :3] = rot_mat
    result[:, :3, 3] = translations
    result[:, 3, 3] = 1
    return result

def transform_vectors(quaternions, translations, vectors, device=None):
    """
        Args:
            quaternions: quaternions of shape (N, 4)
            translations: translational vectors of shape (N, 3)
            vectors: vectors to be transformed of shape (N, m, 3)

        Returns:
            transformed vectors of shape (N, m, 3)
    """
    tf_matrices = tf_matrices_from_poses(translations, quaternions, device)
    return transform_vectors_from_matrix(tf_matrices, vectors, device)

def inverse_transform_vectors(quaternions, translations, vectors, device=None):
    """
        Args:
            quaternions: quaternions of shape (N, 4)
            translations: translational vectors of shape (N, 3)
            vectors: vectors to be transformed of shape (N, m, 3)

        Returns:
            inversely transformed vectors of shape (N, m, 3)
    """
    tf_matrices = tf_matrices_from_poses(translations, quaternions, device)
    inverse_tf_matrices = inverse_transform_matrices(tf_matrices, device)
    return transform_vectors_from_matrix(inverse_tf_matrices, vectors, device)

def transform_vectors_from_matrix(tf_matrices, vectors, device=None):
    """
        Args:
            tf_matrice: transformation matrices of shape (N, 4, 4)
            vectors: vectors to be transformed of shape (N, m, 3)

        Returns:
            transformed vectors of shape (N, m, 3)
    """
    # Transform vector into shape (N, 4, m)
    transposed_vector = torch.transpose(vectors, 1, 2) # (N, 3, m)
    homogeneous_vectors = torch.ones((transposed_vector.shape[0], 4, transposed_vector.shape[2]), 
            dtype=torch.float32, 
            device=device)
    homogeneous_vectors[:, 0:3, :] = transposed_vector

    # Apply batch matrix multiplication
    transformed_homo_vectors = torch.bmm(tf_matrices, homogeneous_vectors)
    transformed_transposed_vectors = transformed_homo_vectors[:, 0:3, :] # (N, 3, m)
    transformed_vectors = torch.transpose(transformed_transposed_vectors, 1, 2)

    return transformed_vectors
    
def inverse_transform_matrices(tf_matrices, device=None):
    """
        Get the inverse matrices of the transformation matrices

        Args:
            tf_matrices: transformation matrices to be inverse of shape (N, 4, 4)

        Return:
            Inverse of the transformation matrices of shape (N, 4, 4)
    """
    rot_matrices = tf_matrices[:, 0:3, 0:3]
    inverse_rot_matrices = torch.transpose(rot_matrices, 1, 2) # (N, 3, 3)

    minus_p_vector = -tf_matrices[:, 0:3, 3].view(tf_matrices.shape[0], 3, 1) # (N, 3, 1)
    inversed_p_vector = torch.bmm(inverse_rot_matrices, minus_p_vector)

    inverse_tf_matrices = torch.zeros((tf_matrices.shape[0], 4, 4), dtype=torch.float32, device=device)
    inverse_tf_matrices[:, 0:3, 0:3] = inverse_rot_matrices
    inverse_tf_matrices[:, 0:3, 3] = inversed_p_vector.view(tf_matrices.shape[0], 3)
    inverse_tf_matrices[:, 3, 3] = 1.0

    return inverse_tf_matrices

def get_pose_in_spherical_coord(sphere_quaternion_in_robot, 
                                sphere_position_in_robot, 
                                ball_radius):
    device = sphere_quaternion_in_robot.device
    num_envs = sphere_quaternion_in_robot.shape[0]
    robot_position_in_ball = quat_rotate_inverse(sphere_quaternion_in_robot, -sphere_position_in_robot)
    robot_quaternion_in_ball = quat_conjugate(sphere_quaternion_in_robot)

    x = robot_position_in_ball[:, 0]
    y = robot_position_in_ball[:, 1]
    z = robot_position_in_ball[:, 2]
    r = torch.norm(robot_position_in_ball, p=2, dim=-1)
    theta = torch.acos(z/r)
    phi = torch.atan2(y, x)
    ## Get the virtual position vector
    virtual_x = (theta - torch.pi/2) * r
    virtual_y = phi * r
    virtual_z = r - ball_radius
    ## Get the local orthonormal unit vectors
    e_r_x = torch.sin(theta) * torch.cos(phi)
    e_r_y = torch.sin(theta) * torch.sin(phi)
    e_r_z = torch.cos(theta)
    e_theta_x = torch.cos(theta) * torch.cos(phi)
    e_theta_y = torch.cos(theta) * torch.sin(phi)
    e_theta_z = -torch.sin(theta)
    e_phi_x = -torch.sin(phi)
    e_phi_y = torch.cos(phi)
    e_phi_z = torch.zeros_like(e_phi_y)
    ## Get the rotation matrix of local orthonormal unit vectors relative to the ball frame
    rot_mat_local_frame = torch.zeros(num_envs, 3, 3, dtype=torch.float32, device=device)
    rot_mat_local_frame[:, 0, 0] = e_theta_x
    rot_mat_local_frame[:, 1, 0] = e_theta_y
    rot_mat_local_frame[:, 2, 0] = e_theta_z
    rot_mat_local_frame[:, 0, 1] = e_phi_x
    rot_mat_local_frame[:, 1, 1] = e_phi_y
    rot_mat_local_frame[:, 2, 1] = e_phi_z
    rot_mat_local_frame[:, 0, 2] = e_r_x
    rot_mat_local_frame[:, 1, 2] = e_r_y
    rot_mat_local_frame[:, 2, 2] = e_r_z
    local_frame_quat_in_ball = quat_from_rot_matrices(rot_mat_local_frame, device)
    ## Get the virtual robot orientation
    virtual_robot_quat_in_local_frame = inverse_rotate_orientations(local_frame_quat_in_ball, 
                                                                    robot_quaternion_in_ball,
                                                                    device)
    
    return virtual_x, virtual_y, virtual_z, virtual_robot_quat_in_local_frame

# @torch.jit.script
def rand_quaternions(num_envs: int, 
                     min_roll: float,
                     max_roll: float,
                     min_pitch: float, 
                     max_pitch: float,
                     min_yaw: float, 
                     max_yaw: float,
                     device: str) -> torch.Tensor:
    rand_values = torch.rand((num_envs, 3), device=device) # [0, 1)

    # Range of euler angles from [min, max]
    rand_roll = min_roll + (max_roll-min_roll) * rand_values[:, 0]
    rand_pitch = min_pitch + (max_pitch-min_pitch) * rand_values[:, 1]
    rand_yaw = min_yaw + (max_yaw-min_yaw) * rand_values[:, 2]

    rand_quaternion = quat_from_euler_xyz(rand_roll, rand_pitch, rand_yaw)

    return rand_quaternion

@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps:float=2) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())

if __name__ == "__main__":
    device = 'cpu'
    # vectors = torch.tensor([[[1,1,0], [1,1,1]],[[2,2,0],[2,2,2]]], device=device)
    # translations = torch.tensor([1,1,1], dtype=torch.float32).repeat((2,1))
    # quaternions = torch.tensor([1,0,0,0], dtype=torch.float32).repeat(2, 1)

    # result_vectors = transform_vectors(quaternions, translations, vectors, device)
    
    # print(result_vectors)

    # test_matrices = torch.tensor([[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]]).repeat((3, 1, 1))

    # print(test_matrices)
    # print(inverse_transform_matrices(test_matrices))

    # test_rot_mat = torch.tensor([[0,0,-1],[-1,0,0],[0,1,0]]).repeat((2, 1, 1)).view(2, 3, 3)
    # result_quat = quat_from_rot_matrices(test_rot_mat)
    # print(result_quat)

    test_rot_quat = torch.tensor([-0.5, -0.5, 0.5, 0.5]).repeat(2, 1)
    test_quat = torch.tensor([0.7071, 0, 0, 0.7071]).repeat(2, 1)
    result_quat = inverse_rotate_orientations(test_rot_quat, test_quat)

    print(result_quat)


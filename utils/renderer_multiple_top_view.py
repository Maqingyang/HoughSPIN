import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, faces=None, img=None, ):
        # self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
        #                                viewport_height=img_res,
        #                                point_size=1.0)
        H, W = img.shape[0], img.shape[1]


        self.focal_length = focal_length
        self.camera_center = [W // 2, H // 2]
        # self.camera_center = [500, 500]
        self.faces = faces

        # self.renderer = pyrender.OffscreenRenderer(viewport_width=W,
        #                     viewport_height=H,
        #                     point_size=1.0)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=2000,
                            viewport_height=2000,
                            point_size=1.0)

    def __call__(self, verts, image, mesh_trans, rotate_theta=0):


        colors = [(0.8, 0.3, 0.3, 1.0)]
        num_people = verts.shape[0]

        
        # verts is (N, 6890, 3)
        # trans is (N, 3), the translation for each mesh
        camera_center = np.array([image.shape[1] / 2., image.shape[0] / 2.]) 
        for n in range(num_people):
            mesh_trans[n,:2] -=  camera_center * mesh_trans[n,2] / self.focal_length
        verts = verts + mesh_trans[:, None, ]
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.5, 0.5, 0.5))
        # Create camera. Camera will always be at [0,0,0]
        camera_pose = np.eye(4)     
        if True:
            half_depth = (mesh_trans[:,2].max() + mesh_trans[:,2].min())/2.
            half_point = np.array([0,0,-half_depth])
            R = np.zeros((3,3))
            R[0,0] = 1
            R[1,1] = np.cos(rotate_theta)
            R[1,2] = np.sin(rotate_theta)
            R[2,1] = -np.sin(rotate_theta)
            R[2,2] = np.cos(rotate_theta)
            T = np.array([0,
                          half_depth*np.sin(rotate_theta),
                          -half_depth*(1-np.cos(rotate_theta)),
                        ])
            camera_pose[:3,:3] = R
            camera_pose[:3, 3] = T


        # camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
        #                                         cx=camera_center[0], cy=camera_center[1])
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                cx=1000, cy=1000)
        scene.add(camera, pose=camera_pose)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        for n in range(num_people):

            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.2,
                            alphaMode='OPAQUE',
                            baseColorFactor=colors[n % len(colors)])
            mesh = pyrender.Mesh.from_trimesh(
                            mesh,
                            material=material)
            scene.add(mesh, 'mesh')


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        image = np.ones((2000,2000,3))
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        
        
        return output_img


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
    def __init__(self, focal_length=5000, img_res=224, faces=None, img=None):
        # self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
        #                                viewport_height=img_res,
        #                                point_size=1.0)
        H, W = img.shape[0], img.shape[1]

        self.renderer = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [W // 2, H // 2]
        # self.camera_center = [500, 500]
        self.faces = faces


    def __call__(self, verts, image, mesh_trans):


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
        # camera_center *= 1.6
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                cx=camera_center[0], cy=camera_center[1])

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

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        
        
        return output_img


      # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(0.8, 0.3, 0.3, 1.0))
        # mesh = trimesh.Trimesh(vertices, self.faces)
        # rot = trimesh.transformations.rotation_matrix(
        #     np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        # scene.add(mesh, 'mesh')

        # camera_translation[0] *= -1.
        # camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        # camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
        #                                    cx=self.camera_center[0], cy=self.camera_center[1])
        # scene.add(camera, pose=camera_pose)
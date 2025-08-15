import numpy as np
from cameras.cameras import Camera
from coremaths.vector import Vec3
from coremaths.frame import Frame
from coremaths.geometry import Rectangle, Spheroid
from rendering.renderables import RenderableObject, RenderableScene
from rendering.lights import Light
from rendering.renderer import Renderer
from rendering.meshes import Mesh
from rendering.textures import Texture
from radiometry.reflectance_funcs import BRDF, TexturedBRDF as tBRDF
from simply_utils import paths, constants as consts
import matplotlib.pyplot as plt


def test_fullRenderingFlow():
    """Tests that the full SIMply rendering flow - from scene definition to image generation - runs without error and
    produces correct images (by comparing to reference images)."""
    meshPath = paths.dataFilePath(['input', 'examples', 'model1'], 'surface.obj')
    mesh = Mesh.loadFromOBJ(meshPath)

    mesh.textureCoordArray = None
    mesh.triTexIndices = None

    brdfPath = paths.dataFilePath(['input', 'examples', 'model1'], 'lambert_ref.npy')
    brdf_vals = np.load(brdfPath)
    brdf_tex = Texture.planetocentric(brdf_vals, (0, 360, -90, 90))
    brdf = tBRDF(BRDF.lambert, (brdf_tex,))

    texPath = paths.dataFilePath(['input', 'examples', 'model1'], 'texture.jpg')
    tex = Texture.planetocentric(texPath, (0, 360, -90, 90))

    renderable = RenderableObject.renderableMesh(mesh, brdf, tex)

    au = consts.au
    light = Light.sunPointSource(au * Vec3((3, 2, 1)).norm)
    scene = RenderableScene([renderable], light)

    cam = Camera.pinhole((40, 40), 500, 500)
    cam.psfSigma = 1
    cam.psfType = 'airy'
    cam.binning = 2
    cam.nr = 25

    cam.frame = Frame.withW(Vec3((0, -1, 0)), origin=Vec3((0, 400, 0)))

    img, radiance = Renderer.image(scene, cam, 0.01, [500, 501, 502], sf=2, n_shad=1)
    tex_img = Renderer.texture(scene, cam)

    path = paths.dataFilePath(['test_data', 'render'], 'radiance.npy')
    ref_radiance = np.load(path)
    path = paths.dataFilePath(['test_data', 'render'], 'dn_image.npy')
    ref_img = np.load(path)
    path = paths.dataFilePath(['test_data', 'render'], 'texture_image.npy')
    ref_tex_img = np.load(path)

    assert np.array_equal(ref_radiance, radiance)
    assert np.allclose(ref_img, img, atol=5, rtol=0)
    assert np.array_equal(ref_tex_img, tex_img, equal_nan=True)

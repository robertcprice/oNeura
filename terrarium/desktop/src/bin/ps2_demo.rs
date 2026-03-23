use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

use bevy::window::WindowResolution;
use std::f32::consts::PI;

const PS2_WIDTH: u32 = 320;
const PS2_HEIGHT: u32 = 240;

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct PixelCamera;

#[derive(Component)]
struct TargetObject {
    name: String,
    structure_type: String,
}

#[derive(Component)]
struct MacroGroup;

#[derive(Component)]
struct MolecularGroup;

#[derive(States, Default, Debug, Clone, Eq, PartialEq, Hash)]
enum ScaleMode {
    #[default]
    Macro,
    Molecular,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(800.0, 600.0),
                title: "oNeura PS2 Terrarium Demo".into(),
                ..default()
            }),
            ..default()
        }).set(ImagePlugin::default_nearest()))
        .init_state::<ScaleMode>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            camera_controls,
            animate_fish,
            check_zoom_level,
            update_visibility,
        ))
        .run();
}

#[derive(Resource)]
struct ResolutionSettings {
    render_image: Handle<Image>,
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
) {
    // 1. Setup the low-res render target texture
    let size = Extent3d {
        width: PS2_WIDTH,
        height: PS2_HEIGHT,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        sampler: bevy::image::ImageSampler::nearest(),
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);
    commands.insert_resource(ResolutionSettings {
        render_image: image_handle.clone(),
    });

    // 2. Spawn the pixel camera (renders to the low-res texture)
    commands.spawn((
        Camera3d::default(),
        Camera {
            target: RenderTarget::Image(image_handle.clone()),
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // 3. Spawn the window camera (renders the low-res texture to screen)
    commands.spawn((
        Camera2d,
        PixelCamera,
    ));
    commands.spawn((
        Sprite {
            image: image_handle,
            custom_size: Some(Vec2::new(800.0, 600.0)),
            ..default()
        },
    ));

    // Lights
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-4.0, 5.0, -4.0),
    ));

    // Macro Scene
    let macro_parent = commands.spawn((
        Transform::default(),
        Visibility::Visible,
        MacroGroup,
    )).id();

    // Soil
    let soil = commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(10.0, 1.0, 10.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.33, 0.20, 0.07),
            unlit: false,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.5, 0.0),
        TargetObject { name: "Soil".into(), structure_type: "soil".into() },
    )).id();

    // Water
    let water = commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(4.0, 0.5, 4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.13, 0.4, 0.8, 0.8),
            alpha_mode: AlphaMode::Blend,
            ..default()
        })),
        Transform::from_xyz(2.0, 0.25, 2.0),
        TargetObject { name: "Water".into(), structure_type: "water".into() },
    )).id();

    // Plant
    let plant = commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.2, 3.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.66, 0.2),
            ..default()
        })),
        Transform::from_xyz(-2.0, 1.5, -2.0),
        TargetObject { name: "Plant".into(), structure_type: "plant".into() },
    )).id();

    // Fish
    let fish = commands.spawn((
        Mesh3d(meshes.add(Cone { radius: 0.3, height: 1.0 })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.4, 0.0),
            ..default()
        })),
        Transform::from_xyz(2.0, 0.5, 2.0).with_rotation(Quat::from_rotation_z(PI/2.0)),
        TargetObject { name: "Fish".into(), structure_type: "fish".into() },
        FishAnim,
    )).id();

    commands.entity(macro_parent).add_children(&[soil, water, plant, fish]);

    // Molecular Scene (Hidden initially)
    let mol_parent = commands.spawn((
        Transform::default(),
        Visibility::Hidden,
        MolecularGroup,
    )).id();

    // Just spawn some placeholder atoms to represent molecular zoom
    let atom_mesh = meshes.add(Sphere::new(0.1).mesh().build());
    let bond_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.5),
        ..default()
    });
    
    // We'll spawn a mini molecule cluster at the center
    for i in 0..10 {
        let angle = (i as f32) * PI * 2.0 / 10.0;
        let x = angle.cos() * 0.5;
        let y = (i as f32 * 0.1) - 0.5;
        let z = angle.sin() * 0.5;
        
        let color = if i % 2 == 0 { Color::srgb(0.8, 0.2, 0.2) } else { Color::srgb(0.2, 0.2, 0.8) };
        let mat = materials.add(StandardMaterial {
            base_color: color,
            ..default()
        });

        let atom = commands.spawn((
            Mesh3d(atom_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(x, y, z),
        )).id();
        commands.entity(mol_parent).add_child(atom);
    }
}

#[derive(Component)]
struct FishAnim;

fn animate_fish(time: Res<Time>, mut q: Query<&mut Transform, With<FishAnim>>) {
    for mut transform in q.iter_mut() {
        let t = time.elapsed_secs();
        transform.translation.x = 2.0 + (t * 1.5).sin() * 1.5;
        transform.translation.z = 2.0 + (t * 1.5).cos() * 1.5;
        transform.rotation = Quat::from_rotation_y(-(t * 1.5) + PI) * Quat::from_rotation_z(PI/2.0);
    }
}

fn camera_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut q_cam: Query<&mut Transform, With<MainCamera>>,
) {
    if let Ok(mut transform) = q_cam.get_single_mut() {
        let speed = 5.0 * time.delta_secs();
        let rot_speed = 1.0 * time.delta_secs();
        
        let mut forward = transform.forward().as_vec3();
        let mut right = transform.right().as_vec3();
        
        if keys.pressed(KeyCode::KeyW) {
            transform.translation += forward * speed;
        }
        if keys.pressed(KeyCode::KeyS) {
            transform.translation -= forward * speed;
        }
        if keys.pressed(KeyCode::KeyA) {
            transform.translation -= right * speed;
        }
        if keys.pressed(KeyCode::KeyD) {
            transform.translation += right * speed;
        }
        
        if keys.pressed(KeyCode::KeyQ) {
            transform.rotate_y(rot_speed);
        }
        if keys.pressed(KeyCode::KeyE) {
            transform.rotate_y(-rot_speed);
        }
    }
}

fn check_zoom_level(
    q_cam: Query<&Transform, With<MainCamera>>,
    mut state: ResMut<NextState<ScaleMode>>,
    current_state: Res<State<ScaleMode>>,
) {
    if let Ok(cam_t) = q_cam.get_single() {
        // If camera gets very close to center (where molecular view is placed for demo)
        // In a real app we'd raycast to the specific object
        let dist = cam_t.translation.length();
        if dist < 2.0 && *current_state.get() == ScaleMode::Macro {
            state.set(ScaleMode::Molecular);
            println!("Zoomed into Molecular Structure!");
        } else if dist >= 2.0 && *current_state.get() == ScaleMode::Molecular {
            state.set(ScaleMode::Macro);
            println!("Zoomed out to Macro Ecosystem!");
        }
    }
}

fn update_visibility(
    state: Res<State<ScaleMode>>,
    mut q_macro: Query<&mut Visibility, (With<MacroGroup>, Without<MolecularGroup>)>,
    mut q_mol: Query<&mut Visibility, (With<MolecularGroup>, Without<MacroGroup>)>,
) {
    if state.is_changed() {
        match state.get() {
            ScaleMode::Macro => {
                for mut vis in q_macro.iter_mut() { *vis = Visibility::Visible; }
                for mut vis in q_mol.iter_mut() { *vis = Visibility::Hidden; }
            }
            ScaleMode::Molecular => {
                for mut vis in q_macro.iter_mut() { *vis = Visibility::Hidden; }
                for mut vis in q_mol.iter_mut() { *vis = Visibility::Visible; }
            }
        }
    }
}

# Setting up subwidgets for the master widget
import napari
import numpy as np
import torch
from magicgui.widgets import Container, PushButton, create_widget

from w_seg_napari._wsegutils import process_large_image


class WNetWidget(Container):
    def __init__(self, viewer: 'napari.Viewer', generator=None, device=None):
        super().__init__()
        self.viewer = viewer
        self.generator = generator
        self.device = device
        self.live_update_layer = None
        self.output_layer = None
        # Create GUI fields
        self.image_layer_select = create_widget(
            annotation='napari.layers.Image', label='Image Layer'
        )
        self.patch_size_input = create_widget(
            value=256, annotation=int, label='Patch Size'
        )
        self.stride_input = create_widget(
            value=128, annotation=int, label='Stride'
        )
        self.mode_input = create_widget(
            value='Live View',
            widget_type='RadioButtons',
            options={'choices': ['Normal (Fast)', 'Live View']},
            label='Execution Mode:',
        )
        # Set up push button
        self.predict_btn = PushButton(label='Predict Matrix Output')
        # Connect button click
        self.predict_btn.changed.connect(self._on_predict)

        # Build widget layout
        self.extend(
            [
                self.image_layer_select,
                self.patch_size_input,
                self.stride_input,
                self.mode_input,
                self.predict_btn,
            ]
        )

        # Layer updater
        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(
                self._update_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self._update_layer_choices
            )
            # Perform initial populating of layer choices
            self._update_layer_choices()

    def _update_layer_choices(self, event=None):
        """Refreshes the dropdown choices whenever layers are added or removed."""
        if hasattr(self.image_layer_select, 'reset_choices'):
            self.image_layer_select.reset_choices()

    def _on_predict(self):
        """Callback executed when clicking the predict button."""
        image_layer = self.image_layer_select.value
        if image_layer is None:
            # Fallback to active layer if none selected in dropdown
            active = self.viewer.layers.selection.active
            if isinstance(active, napari.layers.Image):
                image_layer = active

        if image_layer is None:
            print('[w-seg] Error: Please select an Image layer.')
            return

        if self.generator is None:
            print('[w-seg] Error: Model generator missing.')
            return
        # Setup input parameters
        patch_size = self.patch_size_input.value
        stride = self.stride_input.value
        is_live = self.mode_input.value == 'Live View'
        # Print device
        print(f'[w-seg] Running prediction with model on {self.device}...')

        raw_data = image_layer.data
        H, W = raw_data.shape[0], raw_data.shape[1]
        # convert image to 0-1
        if raw_data.dtype == np.uint8:
            raw_data = raw_data.astype(np.float32) / 255.0
        else:
            raw_data = raw_data.astype(np.float32)

        input_tensor = torch.from_numpy(raw_data).permute(2, 0, 1).float()

        # Initialize empty canvas. If Normal mode, keep it completely transparent (0.0) until the end
        canvas = np.zeros((H, W), dtype=np.float32)
        if self.live_update_layer in self.viewer.layers:
            self.live_update_layer.data = canvas
            self.live_update_layer.opacity = 0.8 if is_live else 0.0
        else:
            self.live_update_layer = self.viewer.add_image(
                canvas,
                name='live_update_layer',
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=0.8 if is_live else 0.0,
            )

        print(
            f'[w-seg] Stitching processing initialized using patches of size {patch_size} [{self.mode_input.value}]...'
        )

        final_sem, final_aff = process_large_image(
            image=input_tensor,
            model=self.generator,
            patch_size=patch_size,
            stride=stride,
            device=self.device,
            live_layer=self.live_update_layer,
            live_view=is_live,
        )

        # Final full frame rendering and resolution
        # sem_numpy = np.clip(sem_numpy, 0.0, 1.0)
        sem_numpy = np.squeeze(final_sem)  # Ensure it's 2D
        aff_numpy = np.clip(final_aff, 0.0, 1.0)

        # Grab the input layer's name and format it with the 'pred_' prefix
        original_name = image_layer.name
        predicted_layer_name = f'pred_{original_name}'

        # Add affininity layers to viewer
        aff_layers = self.viewer.add_image(
            aff_numpy,
            name=[f'affH_{original_name}', f'affV_{original_name}'],
            channel_axis=0,  # Splits 2 channel image (C, H, W)
            colormap='gray',
            blending='additive',
        )
        # Move the affinity layers to the bottom of the layer stack for better visibility
        for layer in aff_layers:
            # Move layer to index 0. Doing this twice moves both to the bottom.
            current_idx = self.viewer.layers.index(layer)
            self.viewer.layers.move(current_idx, 0)

        # Remove live layer and commit the processed values to the layer structure and reveal it cleanly
        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.live_update_layer)
            self.live_update_layer = None  # Reset tracking reference
        if predicted_layer_name in self.viewer.layers:
            # Overwrite data in existing layer
            final_layer = self.viewer.layers[predicted_layer_name]
            final_layer.data = sem_numpy  # or final_sem.squeeze(0)
            final_layer.refresh()
        else:
            final_layer = self.viewer.add_image(
                sem_numpy,  # or final_sem.squeeze(0)
                name=predicted_layer_name,
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=1.0,
            )

        # Deselect all other layers and select only the live update layer
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(final_layer)

        # Mak sure that semantic layer is on top of the affinity layers for better visibility
        sem_idx = self.viewer.layers.index(final_layer)
        self.viewer.layers.move(sem_idx, -1)

        print(
            '[w-seg] Processing complete! Final canvas layers blended successfully.'
        )

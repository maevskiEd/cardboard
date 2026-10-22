/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HELLO_CARDBOARD_ANDROID_SRC_MAIN_JNI_HELLO_CARDBOARD_APP_H_
#define HELLO_CARDBOARD_ANDROID_SRC_MAIN_JNI_HELLO_CARDBOARD_APP_H_

#include <android/asset_manager.h>
#include <jni.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <GLES2/gl2.h>
#include "cardboard.h"
#include "util.h"

namespace ndk_hello_cardboard {

/**
 * This is a sample app for the Cardboard SDK. It loads a simple environment and
 * objects that you can click on.
 */
class HelloCardboardApp {
 public:
  /**
   * Creates a HelloCardboardApp.
   *
   * @param vm JavaVM pointer.
   * @param obj Android activity object.
   * @param asset_mgr_obj The asset manager object.
   */
  HelloCardboardApp(JavaVM* vm, jobject obj, jobject asset_mgr_obj);

  ~HelloCardboardApp();

  /**
   * Initializes any GL-related objects. This should be called on the rendering
   * thread with a valid GL context.
   *
   * @param env The JNI environment.
   */
  void OnSurfaceCreated(JNIEnv* env);

  /**
   * Sets screen parameters.
   *
   * @param width Screen width
   * @param height Screen height
   */
  void SetScreenParams(int width, int height);

  /**
   * Draws the scene. This should be called on the rendering thread.
   */
  void OnDrawFrame();

  /**
   * Hides the target object if it's being targeted.
   */
  void OnTriggerEvent();

  /**
   * Pauses head tracking.
   */
  void OnPause();

  /**
   * Resumes head tracking.
   */
  void OnResume();

  /**
   * Allows user to switch viewer.
   */
  void SwitchViewer();

 private:
  /**
   * Default near clip plane z-axis coordinate.
   */
  static constexpr float kZNear = 0.1f;

  /**
   * Default far clip plane z-axis coordinate.
   */
  static constexpr float kZFar = 100.f;

  /**
   * Updates device parameters, if necessary.
   *
   * @return true if device parameters were successfully updated.
   */
  bool UpdateDeviceParams();

  /**
   * Initializes GL environment.
   */
  void GlSetup();

  /**
   * Deletes GL environment.
   */
  void GlTeardown();

  /**
   * Gets head's pose as a 4x4 matrix.
   *
   * @return matrix containing head's pose.
   */
  Matrix4x4 GetPose();

  /**
   * Draws all world-space objects for the given eye.
   */
  void DrawWorld();

  /**
   * Draws the target object.
   */
  void DrawTarget();

  /**
   * Draws the room.
   */
  void DrawRoom();

  /**
   * Finds a new random position for the target object.
   */
  void HideTarget();

  /**
   * Checks if user is pointing or looking at the target object by calculating
   * whether the angle between the user's gaze and the vector pointing towards
   * the object is lower than some threshold.
   *
   * @return true if the user is pointing at the target object.
   */
  bool IsPointingAtTarget();

  /**
   * What the user's gaze is currently resting on.
   */
  enum class GazeItem { kNone, kTarget, kMagnet, kNewTargetButton, kRecenterButton };

  /**
   * Updates the panel pose, the gaze item and the dwell/detach timers. Must be
   * called once per frame after |head_view_| has been updated.
   *
   * @param dt Seconds since the previous frame.
   */
  void UpdateInteraction(float dt);

  /**
   * Finds the item under the reticle.
   *
   * @param distance Receives the head-space distance to the item, if any.
   * @return The gazed item, or kNone.
   */
  GazeItem FindGazeItem(float* distance);

  /**
   * Checks whether the reticle is inside a square panel element.
   *
   * @param local_x Element center X in panel space, meters.
   * @param local_y Element center Y in panel space, meters.
   * @param half_size Half of the element side, meters.
   * @param distance Receives the head-space distance to the element.
   * @return true if the user is looking at the element.
   */
  bool IsGazingAtPanelElement(float local_x, float local_y, float half_size,
                              float* distance);

  /**
   * Performs the action of a gaze item (dwell finished or trigger pressed).
   */
  void Activate(GazeItem item);

  /**
   * Places the panel in front of the user, keeping it upright.
   */
  void PlacePanelInFront();

  /**
   * Attaches the panel to the head so it follows head rotation.
   */
  void AttachPanel();

  /**
   * Leaves the panel at its current place in the world.
   */
  void DetachPanel();

  /**
   * Gets the direction the head is looking at, in world space.
   */
  std::array<float, 4> GetHeadForward();

  /**
   * Draws the panel with its buttons for the current eye.
   */
  void DrawPanel(Matrix4x4 eye_view, Matrix4x4 projection);

  /**
   * Draws the reticle with the dwell/detach progress ring for the current eye.
   */
  void DrawReticle(Matrix4x4 eye_matrix, Matrix4x4 projection);

  /**
   * Draws a unit quad with the UI shader.
   */
  void DrawUiQuad(Matrix4x4 mvp, const std::array<float, 4>& color,
                  float icon, float width, float height, float progress);

  jobject java_asset_mgr_;
  AAssetManager* asset_mgr_;

  CardboardHeadTracker* head_tracker_;
  CardboardLensDistortion* lens_distortion_;
  CardboardDistortionRenderer* distortion_renderer_;

  CardboardEyeTextureDescription left_eye_texture_description_;
  CardboardEyeTextureDescription right_eye_texture_description_;

  bool screen_params_changed_;
  bool device_params_changed_;
  int screen_width_;
  int screen_height_;

  float projection_matrices_[2][16];
  float eye_matrices_[2][16];

  GLuint depthRenderBuffer_;  // depth buffer
  GLuint framebuffer_;        // framebuffer object
  GLuint texture_;            // distortion texture

  GLuint obj_program_;
  GLuint obj_position_param_;
  GLuint obj_uv_param_;
  GLuint obj_modelview_projection_param_;

  Matrix4x4 head_view_;
  Matrix4x4 model_target_;

  Matrix4x4 modelview_projection_target_;
  Matrix4x4 modelview_projection_room_;

  TexturedMesh room_;
  Texture room_tex_;

  std::vector<TexturedMesh> target_object_meshes_;
  std::vector<Texture> target_object_not_selected_textures_;
  std::vector<Texture> target_object_selected_textures_;
  int cur_target_object_;

  GLuint ui_program_;
  GLuint ui_position_param_;
  GLuint ui_mvp_param_;
  GLuint ui_color_param_;
  GLuint ui_icon_param_;
  GLuint ui_size_param_;
  GLuint ui_progress_param_;

  // Panel pose in world space. While attached it is recomputed every frame
  // from |panel_in_head_|.
  Matrix4x4 panel_model_;
  Matrix4x4 panel_in_head_;
  bool panel_placed_;
  bool panel_attached_;

  // Dwell click: the item looked at, for how long and whether it already fired
  // (the gaze must leave the item before it can fire again).
  GazeItem gaze_item_;
  float gaze_distance_;
  float dwell_time_;
  bool dwell_fired_;

  // Detach: the head has to turn first, then stay still.
  std::array<float, 4> grab_forward_;
  std::array<float, 4> prev_forward_;
  bool drag_moved_;
  float head_speed_;
  float still_time_;

  int64_t last_frame_time_;
};

}  // namespace ndk_hello_cardboard

#endif  // HELLO_CARDBOARD_ANDROID_SRC_MAIN_JNI_HELLO_CARDBOARD_APP_H_

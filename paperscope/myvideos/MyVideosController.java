package com.google.vr.cardboard.paperscope.demo.myvideos;

import android.os.Handler;
import android.os.Looper;
import com.google.cardboard.sdk.CardboardView;
import com.google.vr.cardboard.paperscope.demo.common.DemoBaseController;

/** Demo that allows the user to view My Videos. */
public class MyVideosController extends DemoBaseController<MyVideosRenderer> {
  private final VideoSequencer sequencer;

  public MyVideosController(CardboardView cardboardView) {
    super(cardboardView);
    renderer = new MyVideosRenderer(cardboardView.getContext());
    sequencer = new VideoSequencer(cardboardView.getContext(), cardboardView, renderer);
    // The dwell fires on the GL thread; the player is driven from the UI thread like the trigger.
    Handler mainHandler = new Handler(Looper.getMainLooper());
    renderer.setThumbnailDwellListener(
        id -> mainHandler.post(() -> sequencer.playThumbnailVideo(id)));
    this.cardboardView.setRenderer(renderer);
  }

  @Override
  public void onResume() {
    super.onResume();
    sequencer.startPlayback();
  }

  @Override
  public void onPause() {
    super.onPause();
    sequencer.stop();
  }

  @Override
  public void shutdown() {
    super.shutdown();
    sequencer.release();
  }

  @Override
  public void enableOnTriggerEventListener() {
    cardboardView.setOnTriggerEvent(
        () -> {
          if (renderer != null && sequencer != null) {
            if (renderer.isScreenOnlyTargeted()) {
              sequencer.togglePause();
            } else {
              sequencer.playTargetedVideo();
            }
          }
        });
  }
}

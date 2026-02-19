"use client";

import { useRef, useEffect, useState } from "react";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { X, BarChartHorizontal } from "lucide-react";

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const normalUrl = useAppStore((state) => state.normalUrl);
  useEffect(() => {
    if (typeof window !== "undefined") {
      console.log("[VideoPlayer] normalUrl:", normalUrl);
    }
  }, [normalUrl]);
  const selectVideo = useAppStore((state) => state.selectVideo);
  const setDebugMode = useAppStore((state) => state.setDebugMode);
  const isPlaying = useAppStore((state) => state.isPlaying);
  const setIsPlaying = useAppStore((state) => state.setIsPlaying);
  const currentTime = useAppStore((state) => state.currentTime);
  const setCurrentTime = useAppStore((state) => state.setCurrentTime);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);

    // Set initial state from store
    video.currentTime = currentTime;
    if (isPlaying) {
      video.play().catch(console.error);
    }

    return () => {
      video.removeEventListener("timeupdate", updateTime);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
    };
  }, [normalUrl]);

  const handleClose = () => {
    selectVideo(null);
  };

  if (!normalUrl) return null;

  return (
    <div className="max-w-6xl mx-auto">
      <Card className="overflow-hidden bg-black/5 dark:bg-black/20">
        <div className="relative aspect-video bg-black">
          <video
            ref={videoRef}
            src={normalUrl}
            className="w-full h-full"
            controls
          />
          <div className="absolute top-4 left-4 z-10">
            <Button
              variant="secondary"
              size="icon"
              className="bg-black/50 hover:bg-black/70"
              aria-label="Show stats"
              onClick={() => setDebugMode(true)}
            >
              <BarChartHorizontal className="h-4 w-4" />
            </Button>
          </div>
          <div className="absolute top-4 right-4 z-10">
            <Button
              variant="destructive"
              size="icon"
              onClick={handleClose}
              className="bg-black/50 hover:bg-black/70"
              aria-label="Close player"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Video, ListVideo, Loader2, AlertTriangle } from "lucide-react";

interface VideoFile {
  path: string;
  name: string;
  uploaded?: boolean;
  video_id?: number;
  normal_url?: string;
  debug_url?: string;
  debug_path?: string;
  debug_json_url?: string;
}

export function VideoSelector() {
  const [libraryVideos, setLibraryVideos] = useState<VideoFile[]>([]);
  const [uploadsVideos, setUploadsVideos] = useState<VideoFile[]>([]);
  const [filteredVideos, setFilteredVideos] = useState<VideoFile[]>([]);
  const [activeTab, setActiveTab] = useState<'library'|'uploads'>('library');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const selectVideo = useAppStore((state) => state.selectVideo);
  const addFrameData = useAppStore((state) => state.addFrameData);
  const clearFrameData = useAppStore((state) => state.clearFrameData);
  const [extracting, setExtracting] = useState<string | null>(null);
  const [promptVideo, setPromptVideo] = useState<VideoFile | null>(null);
  const [promptMinutes, setPromptMinutes] = useState<string>("0.5");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    async function fetchVideos() {
      try {
        setLoading(true);
        setError(null);
        const [videosResp, uploadsResp] = await Promise.all([
          fetch("http://localhost:8000/api/videos"),
          fetch("http://localhost:8000/api/uploads"),
        ]);
        if (!videosResp.ok) throw new Error(`Failed to fetch videos: ${videosResp.statusText}`);
        if (!uploadsResp.ok) throw new Error(`Failed to fetch uploads: ${uploadsResp.statusText}`);
        const videosData = await videosResp.json();
        const uploadsData = await uploadsResp.json();
        // uploadsData sadece upload edilenler, ana listede gösterilmeyecek
        // Normalize items (backend should return {path,name} but handle raw string paths too)
        function normalize(item: any): VideoFile {
          if (!item) return { path: String(item), name: String(item) };
          if (typeof item === "string") {
            const p = item;
            return { path: p, name: p.split("/").pop() || p };
          }
          const p = item.path || String(item);
          const name = item.name || p.split("/").pop() || p;
          const v: VideoFile = { path: p, name };
          if (item.video_id) v.video_id = item.video_id;
          if (item.normal_url) v.normal_url = item.normal_url;
          if (item.debug_url) v.debug_url = item.debug_url;
          if (item.debug_path) v.debug_path = item.debug_path;
          return v;
        }
        const mainList = videosData.map(normalize);
        const uploadsList = (uploadsData || []).map(normalize).map((v) => ({ ...v, uploaded: true }));
        setLibraryVideos(mainList);
        setUploadsVideos(uploadsList);
        setFilteredVideos(activeTab === 'library' ? mainList : uploadsList);
      } catch (e: any) {
        setError(e.message || "An unknown error occurred while fetching videos.");
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    fetchVideos();
  }, []);

  useEffect(() => {
    const lowercasedFilter = searchTerm.toLowerCase();
    const source = activeTab === 'library' ? libraryVideos : uploadsVideos;
    const filtered = source.filter((video) => video.name.toLowerCase().includes(lowercasedFilter));
    setFilteredVideos(filtered);
  }, [searchTerm, libraryVideos, uploadsVideos, activeTab]);

  const handleSelectVideo = (video: VideoFile) => {
    // If this is an uploaded/processed clip, open it directly for playback
    if (video.uploaded) {
      const normal = video.normal_url || `http://localhost:8000/api/video?path=${encodeURIComponent(video.path)}`;
      const debug = video.debug_url || normal;
      selectVideo({ normalUrl: normal, debugUrl: debug, path: video.path, name: video.name });
      // Load associated debug JSON if present
      if (video.debug_json_url) {
        clearFrameData();
        fetch(video.debug_json_url)
          .then((r) => r.ok ? r.json() : Promise.reject(r.statusText))
          .then((arr) => {
            if (Array.isArray(arr)) {
              arr.forEach((f: any) => addFrameData(f));
            }
          })
          .catch((e) => console.warn("Failed to load debug JSON:", e));
      }
      return;
    }
    // Otherwise open inline prompt to ask minutes for extraction/process
    setPromptMinutes("0.5");
    setPromptVideo(video);
    setStatusMessage(null);
  };

  async function extractClipFromVideo(video: VideoFile, minutes: number) {
    setExtracting(video.path);
    setStatusMessage("Videonuz oluşturuluyor ve yükleniyor...");
    try {
      const resp = await fetch("http://localhost:8000/api/process_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: video.path, minutes }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Processing failed");
      }
      const data = await resp.json();
      // select both normal and debug video URLs
      const newUpload: VideoFile = {
        path: data.normal_path,
        name: data.normal_path.split('/').pop() || video.name,
        uploaded: true,
        video_id: data.video_id,
        normal_url: data.normal_url,
        debug_url: data.debug_url,
        debug_path: data.debug_path,
      };
      setUploadsVideos((s) => [newUpload, ...s]);
      selectVideo({ normalUrl: data.normal_url, debugUrl: data.debug_url, path: data.normal_path, name: newUpload.name });
      setStatusMessage("Videonuz ve debug videosu hazır.");
      setTimeout(() => setPromptVideo(null), 800);
    } catch (e: any) {
      console.error(e);
      setStatusMessage("Processing failed: " + (e.message || e));
    } finally {
      setExtracting(null);
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-3xl mb-2">Select a Match Video</CardTitle>
          <CardDescription className="text-lg">
            Choose a video from the list below to start processing.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="relative">
            <Input
              placeholder="Search videos..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8"
            />
            <ListVideo className="absolute left-2 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          </div>

          <div className="flex gap-2">
            <Button
              variant={activeTab === 'library' ? 'default' : 'ghost'}
              onClick={() => setActiveTab('library')}
            >
              Kütüphane
            </Button>
            <Button
              variant={activeTab === 'uploads' ? 'default' : 'ghost'}
              onClick={() => setActiveTab('uploads')}
            >
              Yüklemeler
            </Button>
          </div>

          <ScrollArea className="h-72 w-full rounded-md border">
            <div className="p-4">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  <p className="ml-2">Loading videos...</p>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center h-full text-destructive">
                  <AlertTriangle className="h-8 w-8 mb-2" />
                  <p className="font-semibold">Error loading videos</p>
                  <p className="text-sm text-center">{error}</p>
                </div>
              ) : filteredVideos.length > 0 ? (
                <ul className="space-y-2">
                  {filteredVideos.map((video) => (
                    <li key={video.path}>
                      <Button
                        variant="ghost"
                        className="w-full justify-start text-left h-auto"
                        onClick={() => handleSelectVideo(video)}
                      >
                        <div className="mr-2 flex-shrink-0 h-8 w-12 overflow-hidden rounded-sm bg-black">
                          <img
                            src={`http://localhost:8000/api/thumbnail?path=${encodeURIComponent(video.path)}`}
                            alt="thumb"
                            className="h-full w-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = "/fallback-thumb.png";
                            }}
                          />
                        </div>
                        <span className="truncate">{video.name}</span>
                        {extracting === video.path ? (
                          <Loader2 className="h-4 w-4 ml-2 animate-spin text-muted-foreground" />
                        ) : null}
                      </Button>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-center text-muted-foreground py-10">
                  <p>No videos found.</p>
                </div>
              )}
            </div>
          </ScrollArea>

          {promptVideo ? (
            <div className="fixed inset-0 z-50 flex items-center justify-center">
              <div className="absolute inset-0 bg-black/40" onClick={() => setPromptVideo(null)} />
              <div className="relative z-10 w-full max-w-md">
                <div className="bg-white dark:bg-slate-900 rounded-lg shadow-lg p-4">
                  <h3 className="text-lg font-semibold mb-2">Kaç dakika olacak?</h3>
                  <p className="text-sm text-muted-foreground mb-3 truncate">{promptVideo.name}</p>
                  <div className="flex items-center gap-2 mb-3">
                    <Input
                      value={promptMinutes}
                      onChange={(e) => setPromptMinutes(e.target.value)}
                      type="number"
                      min="0.05"
                      step="0.1"
                      className="w-32"
                    />
                    <span className="text-sm">dakika</span>
                  </div>
                  {statusMessage ? (
                    <div className="mb-3 text-sm">{statusMessage}</div>
                  ) : null}
                  <div className="flex gap-2 justify-end">
                    <Button
                      variant="secondary"
                      onClick={async () => {
                        // process the full video (not just play)
                        setExtracting(promptVideo.path);
                        setStatusMessage("Tüm video işleniyor...");
                        try {
                          const resp = await fetch("http://localhost:8000/api/process_video", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ path: promptVideo.path }),
                          });
                          if (!resp.ok) throw new Error(await resp.text());
                          const data = await resp.json();
                                              const newUpload: VideoFile = {
                            path: data.normal_path,
                            name: data.normal_path.split('/').pop() || promptVideo.name,
                            uploaded: true,
                            video_id: data.video_id,
                            normal_url: data.normal_url,
                                                debug_url: data.debug_url,
                                                debug_path: data.debug_path,
                                                debug_json_url: data.debug_json_url,
                          };
                          // add to uploads tab and select for playback
                          setUploadsVideos((s) => [newUpload, ...s]);
                          selectVideo({ normalUrl: data.normal_url, debugUrl: data.debug_url, path: data.normal_path, name: newUpload.name });
                          setStatusMessage("Video işlendi ve oynatılıyor.");
                        } catch (e: any) {
                          console.error(e);
                          setStatusMessage("İşleme başarısız: " + (e.message || e));
                        } finally {
                          setExtracting(null);
                          setPromptVideo(null);
                        }
                      }}
                    >
                      Tüm videoyu kullan
                    </Button>
                    <Button
                      onClick={() => {
                        const minutes = parseFloat(promptMinutes || "0");
                        if (isNaN(minutes) || minutes <= 0) {
                          setStatusMessage("Lütfen geçerli bir dakika değeri girin.");
                          return;
                        }
                        extractClipFromVideo(promptVideo, minutes);
                      }}
                    >
                      {extracting === promptVideo.path ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Oluşturuluyor...
                        </>
                      ) : (
                        "Kırp ve Yükle"
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

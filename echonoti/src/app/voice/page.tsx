"use client";

import { useState, useRef, useEffect } from "react";
import {
  Mic,
  MicOff,
  Trash2,
  RotateCcw,
  Volume2,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";

interface TimestampedWord {
  word: string;
  start: number;
  end: number;
}

interface VoiceResponse {
  transcription: string;
  response: string;
  audio: string;
  timestamps: TimestampedWord[];
}

interface ConversationTurn {
  id?: number;
  timestamp?: string;
  user: string;
  assistant: string;
  metadata?: any;
  isPlaying?: boolean;
  currentWordIndex?: number;
}

type RecordingState = "idle" | "recording" | "processing" | "playing";

// API base URL - use environment variable or detect from window location
const getApiBaseUrl = () => {
  if (typeof window !== "undefined") {
    // If accessing from network, use the current host's IP/hostname
    const hostname = window.location.hostname;
    if (hostname !== "localhost" && hostname !== "127.0.0.1") {
      return `http://${hostname}:9004`;
    }
  }
  return "http://localhost:9004";
};

export default function VoicePage() {
  const [apiBaseUrl] = useState(getApiBaseUrl());

  // Detect Safari + Network access
  const [showSafariWarning, setShowSafariWarning] = useState(false);

  // Recording state
  const [recordingState, setRecordingState] =
    useState<RecordingState>("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Conversation state
  const [conversations, setConversations] = useState<ConversationTurn[]>([]);
  const [currentTranscription, setCurrentTranscription] = useState("");
  const [processingStage, setProcessingStage] = useState("");

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Detect Safari + Network access on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
      const isHTTP = window.location.protocol === 'http:';
      const isNetwork = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';

      if (isSafari && isHTTP && isNetwork) {
        setShowSafariWarning(true);
      }
    }
  }, []);

  // Load conversation history on mount
  useEffect(() => {
    loadConversationHistory();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (audioElementRef.current) {
        audioElementRef.current.pause();
      }
    };
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [conversations, currentTranscription]);

  const loadConversationHistory = async () => {
    try {
      const response = await fetch(
        `${apiBaseUrl}/api/conversation/history`
      );
      if (response.ok) {
        const data = await response.json();
        setConversations(data.history || []);
      }
    } catch (error) {
      console.error("Error loading history:", error);
    }
  };

  const clearConversationHistory = async () => {
    try {
      const response = await fetch(
        `${apiBaseUrl}/api/conversation/clear`,
        {
          method: "POST",
        }
      );

      if (response.ok) {
        setConversations([]);
        setCurrentTranscription("");
        setError(null);
      }
    } catch (error) {
      console.error("Error clearing history:", error);
      setError("Failed to clear conversation history");
    }
  };

  const startRecording = async () => {
    try {
      setError(null);

      // Request microphone permission (this will show the permission dialog)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set up audio context for visualization
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      // Start audio level monitoring
      monitorAudioLevel();

      // Set up media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        await processAudio(audioBlob);

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecordingState("recording");
      setCurrentTranscription("");
    } catch (error: any) {
      console.error("Error starting recording:", error);

      // Provide helpful error messages
      let errorMessage = "Could not access microphone.";

      if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        errorMessage = "Microphone access denied. Please allow microphone permissions in your browser settings.";
      } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        errorMessage = "No microphone found. Please connect a microphone and try again.";
      } else if (error.name === "NotSupportedError") {
        // Check if accessing via HTTP on Safari
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        const isHTTP = window.location.protocol === 'http:';
        const isNetwork = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';

        if (isSafari && isHTTP && isNetwork) {
          errorMessage = "Safari requires HTTPS for microphone access over network. Please access from the host machine at http://localhost:9003/voice instead.";
        } else {
          errorMessage = "Audio recording is not supported in your browser. Try Chrome or Firefox.";
        }
      } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
        errorMessage = "Microphone is being used by another application. Please close other apps and try again.";
      }

      setError(errorMessage);
      setRecordingState("idle");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recordingState === "recording") {
      mediaRecorderRef.current.stop();
      setRecordingState("processing");
      setAudioLevel(0);

      // Stop animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      // Close audio context
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    }
  };

  const monitorAudioLevel = () => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

    const checkLevel = () => {
      if (!analyserRef.current) return;

      analyserRef.current.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setAudioLevel(Math.min(average / 255, 1)); // Normalize to 0-1

      animationFrameRef.current = requestAnimationFrame(checkLevel);
    };

    checkLevel();
  };

  const processAudio = async (audioBlob: Blob) => {
    setRecordingState("processing");
    setProcessingStage("Converting audio...");

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");

      setProcessingStage("Transcribing speech...");
      const response = await fetch(`${apiBaseUrl}/api/voice/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || `Server error: ${response.status}`
        );
      }

      setProcessingStage("Processing request...");
      const data: VoiceResponse = await response.json();

      setProcessingStage("Generating response...");
      setCurrentTranscription(data.transcription);

      // Add to conversation
      const newTurn: ConversationTurn = {
        user: data.transcription,
        assistant: data.response,
        isPlaying: true,
        currentWordIndex: -1,
      };

      setConversations((prev) => [...prev, newTurn]);

      // Play audio with timestamp synchronization
      if (data.audio) {
        setRecordingState("playing");
        await playAudioWithTimestamps(
          data.audio,
          data.timestamps,
          conversations.length
        );
      }

      // Reload to get persistent ID
      await loadConversationHistory();

      setRecordingState("idle");
      setProcessingStage("");
      setCurrentTranscription("");
    } catch (error) {
      console.error("Error processing audio:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to process audio. Please try again.";
      setError(errorMessage);
      setRecordingState("idle");
      setProcessingStage("");
    }
  };

  const playAudioWithTimestamps = async (
    audioBase64: string,
    timestamps: TimestampedWord[],
    conversationIndex: number
  ) => {
    return new Promise<void>((resolve) => {
      const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
      audioElementRef.current = audio;

      // Update current word based on playback time
      const updateCurrentWord = () => {
        const currentTime = audio.currentTime;
        const index = timestamps.findIndex(
          (ts) => currentTime >= ts.start && currentTime < ts.end
        );

        setConversations((prev) =>
          prev.map((turn, i) =>
            i === conversationIndex
              ? { ...turn, currentWordIndex: index }
              : turn
          )
        );
      };

      audio.ontimeupdate = updateCurrentWord;

      audio.onended = () => {
        setConversations((prev) =>
          prev.map((turn, i) =>
            i === conversationIndex
              ? { ...turn, isPlaying: false, currentWordIndex: -1 }
              : turn
          )
        );
        resolve();
      };

      audio.onerror = () => {
        setConversations((prev) =>
          prev.map((turn, i) =>
            i === conversationIndex
              ? { ...turn, isPlaying: false, currentWordIndex: -1 }
              : turn
          )
        );
        resolve();
      };

      audio.play().catch((err) => {
        console.error("Audio playback error:", err);
        resolve();
      });
    });
  };

  const renderResponseText = (turn: ConversationTurn, index: number) => {
    const response = turn.assistant;
    const words = response.split(/\s+/);

    if (turn.currentWordIndex === undefined || turn.currentWordIndex === -1) {
      return <p className="text-sm sm:text-base leading-relaxed">{response}</p>;
    }

    return (
      <div className="flex flex-wrap gap-1 text-sm sm:text-base leading-relaxed">
        {words.map((word, wordIndex) => (
          <span
            key={wordIndex}
            className={`transition-all duration-200 ${
              wordIndex === turn.currentWordIndex
                ? "text-accent font-semibold scale-105 bg-accent/10 px-1.5 py-0.5 rounded"
                : wordIndex < (turn.currentWordIndex || 0)
                ? "text-muted-foreground"
                : "text-foreground"
            }`}
          >
            {word}
          </span>
        ))}
      </div>
    );
  };

  const getRecordingButtonClass = () => {
    switch (recordingState) {
      case "recording":
        return "bg-destructive hover:bg-destructive/90 scale-105 shadow-destructive/50 animate-pulse";
      case "processing":
      case "playing":
        return "bg-muted/70 opacity-60";
      default:
        return "bg-primary hover:bg-primary/90 hover:scale-110 active:scale-100 shadow-primary/30";
    }
  };

  const getRecordingButtonIcon = () => {
    switch (recordingState) {
      case "recording":
        return <MicOff className="w-14 h-14 sm:w-12 sm:h-12" />;
      case "processing":
      case "playing":
        return (
          <div className="relative w-14 h-14 sm:w-12 sm:h-12">
            <Mic className="w-14 h-14 sm:w-12 sm:h-12 opacity-30" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-16 h-16 sm:w-14 sm:h-14 border-4 border-foreground/20 border-t-foreground rounded-full animate-spin" />
            </div>
          </div>
        );
      default:
        return <Mic className="w-14 h-14 sm:w-12 sm:h-12" />;
    }
  };

  const isDisabled = recordingState !== "idle" && recordingState !== "recording";

  return (
    <div className="container mx-auto p-3 sm:p-4 flex flex-col h-[calc(100dvh-3.5rem)]">
      <div className="w-full max-w-4xl mx-auto flex-1 flex flex-col gap-3 sm:gap-4">
        {/* Header */}
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 flex-1">
            <h1 className="text-xl sm:text-2xl md:text-3xl font-bold font-headline tracking-tighter truncate">
              Voice Assistant
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground">
              Tap mic to speak
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={loadConversationHistory}
              disabled={isDisabled}
              title="Refresh conversation"
              className="h-9 w-9 p-0"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={isDisabled || conversations.length === 0}
                  title="Clear conversation"
                  className="h-9 w-9 p-0"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Clear Conversation?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete all conversation history. This
                    action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={clearConversationHistory}>
                    Clear All
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>

        {/* Safari Network Warning */}
        {showSafariWarning && (
          <Alert className="text-sm sm:text-base">
            <AlertCircle className="h-4 w-4 sm:h-5 sm:w-5" />
            <AlertDescription className="ml-2">
              <strong>Safari Notice:</strong> Microphone blocked over network. Use Chrome/Firefox or access at <strong>localhost:9003/voice</strong> on host machine.
            </AlertDescription>
          </Alert>
        )}

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="text-sm sm:text-base">
            <AlertCircle className="h-4 w-4 sm:h-5 sm:w-5" />
            <AlertDescription className="ml-2">{error}</AlertDescription>
          </Alert>
        )}

        {/* Conversation History */}
        <Card className="flex-1 flex flex-col overflow-hidden bg-card/50 border-primary/20">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {conversations.length === 0 && !currentTranscription && (
                <div className="flex flex-col items-center justify-center h-full py-12 text-center text-muted-foreground px-4">
                  <Volume2 className="w-16 h-16 sm:w-12 sm:h-12 mb-4 opacity-50" />
                  <p className="text-lg sm:text-xl font-semibold">No conversations yet</p>
                  <p className="text-sm sm:text-base mt-2">
                    Tap the microphone to start speaking
                  </p>
                </div>
              )}

              {conversations.map((turn, index) => (
                <div key={turn.id || index} className="space-y-3 sm:space-y-4">
                  {/* User Message */}
                  <div className="flex items-start gap-2 sm:gap-3 justify-end">
                    <div className="max-w-[85%] sm:max-w-[80%] bg-primary/90 text-primary-foreground rounded-lg px-3 py-2.5 sm:px-4 sm:py-3">
                      <p className="text-sm sm:text-base leading-relaxed">{turn.user}</p>
                    </div>
                    <Avatar className="h-9 w-9 sm:h-8 sm:w-8 border-2 border-primary flex-shrink-0">
                      <AvatarFallback className="bg-primary text-primary-foreground text-sm">
                        U
                      </AvatarFallback>
                    </Avatar>
                  </div>

                  {/* Assistant Message */}
                  <div className="flex items-start gap-2 sm:gap-3">
                    <Avatar className="h-9 w-9 sm:h-8 sm:w-8 border-2 border-accent flex-shrink-0">
                      <AvatarFallback className="bg-accent text-accent-foreground text-sm">
                        A
                      </AvatarFallback>
                    </Avatar>
                    <div className="max-w-[85%] sm:max-w-[80%] bg-secondary rounded-lg px-3 py-2.5 sm:px-4 sm:py-3">
                      {turn.isPlaying && (
                        <div className="flex items-center gap-2 mb-2 text-accent">
                          <Volume2 className="w-4 h-4 animate-pulse" />
                          <span className="text-xs sm:text-sm font-medium">Speaking...</span>
                        </div>
                      )}
                      {renderResponseText(turn, index)}
                    </div>
                  </div>
                </div>
              ))}

              {/* Current transcription (temporary) */}
              {currentTranscription && recordingState !== "idle" && (
                <div className="flex items-start gap-3 justify-end opacity-70">
                  <div className="max-w-[80%] bg-primary/90 text-primary-foreground rounded-lg px-4 py-2">
                    <p className="text-sm italic">{currentTranscription}</p>
                  </div>
                  <Avatar className="h-8 w-8 border-2 border-primary">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      U
                    </AvatarFallback>
                  </Avatar>
                </div>
              )}
            </div>
          </ScrollArea>
        </Card>

        {/* Recording Controls */}
        <div className="flex flex-col items-center gap-4 sm:gap-6 pb-4 sm:pb-6">
          {/* Microphone Button */}
          <div className="relative flex flex-col items-center">
            <button
              className={`
                relative rounded-full p-0
                w-28 h-28 sm:w-24 sm:h-24
                flex items-center justify-center
                transition-all duration-300 shadow-2xl
                disabled:cursor-not-allowed
                touch-manipulation
                ${getRecordingButtonClass()}
              `}
              onClick={
                recordingState === "recording" ? stopRecording : startRecording
              }
              disabled={isDisabled}
            >
              {getRecordingButtonIcon()}
            </button>

            {/* Sound wave visualization */}
            {recordingState === "recording" && (
              <div className="mt-6 flex gap-1.5 items-end h-12">
                {[...Array(20)].map((_, i) => {
                  const height =
                    Math.sin((i / 20) * Math.PI * 2 + Date.now() / 100) *
                      audioLevel *
                      35 +
                    6;
                  return (
                    <div
                      key={i}
                      className="w-1 bg-primary rounded-full transition-all duration-75"
                      style={{ height: `${Math.max(6, height)}px` }}
                    />
                  );
                })}
              </div>
            )}
          </div>

          {/* Status Text */}
          <div className="text-center min-h-[4rem] flex flex-col items-center justify-center gap-1 px-4">
            {recordingState === "recording" && (
              <>
                <p className="text-base sm:text-lg font-bold text-destructive">Recording...</p>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Tap to stop
                </p>
              </>
            )}

            {recordingState === "processing" && (
              <div className="flex flex-col items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce" />
                  <div
                    className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "0.15s" }}
                  />
                  <div
                    className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce"
                    style={{ animationDelay: "0.3s" }}
                  />
                </div>
                <p className="text-sm sm:text-base font-medium text-foreground">
                  {processingStage}
                </p>
              </div>
            )}

            {recordingState === "playing" && (
              <>
                <p className="text-base sm:text-lg font-bold text-accent">
                  Playing response
                </p>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Listen carefully
                </p>
              </>
            )}

            {recordingState === "idle" && (
              <p className="text-sm sm:text-base text-muted-foreground">
                {conversations.length === 0
                  ? "Tap the microphone to start"
                  : "Ready for your next question"}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

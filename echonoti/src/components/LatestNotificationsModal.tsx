"use client";

import { useState, useEffect } from "react";
import type { Notification } from "@/lib/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Bell, Check } from "lucide-react";

export function LatestNotificationsModal({ notifications }: { notifications: Notification[] }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  
  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    // Only run this effect on the client side
    if (!isClient) return;
    
    // Only open the modal if there are unread notifications
    if (notifications.length > 0) {
      // Use a session storage item to only show the modal once per session
      const hasBeenShown = sessionStorage.getItem("latestNotificationsModalShown");
      if (!hasBeenShown) {
        setIsOpen(true);
        sessionStorage.setItem("latestNotificationsModalShown", "true");
      }
    }
  }, [notifications, isClient]);

  // Don't render anything until client-side hydration is complete
  if (!isClient || notifications.length === 0) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-[425px] bg-card border-primary/50">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-headline">
            <Bell className="h-5 w-5 text-primary" />
            You have new notifications!
          </DialogTitle>
          <DialogDescription>
            Here are your latest unread items. You can view all items on the main screen.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <ul className="space-y-3">
            {notifications.slice(0, 3).map((notification) => (
              <li key={notification.id} className="text-sm p-3 bg-background/50 rounded-md border">
                <p className="font-medium text-foreground">{notification.headline}</p>
                <p className="text-muted-foreground truncate">{notification.summary}</p>
              </li>
            ))}
          </ul>
        </div>
        <Button onClick={() => setIsOpen(false)} className="w-full">
          <Check className="mr-2 h-4 w-4" /> Got it
        </Button>
      </DialogContent>
    </Dialog>
  );
}

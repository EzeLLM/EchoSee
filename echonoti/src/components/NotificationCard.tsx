"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { format, formatDistanceToNow } from "date-fns";
import { Bookmark, Tag, Trash2 } from "lucide-react";
import type { Notification } from "@/lib/types";
import { cn } from "@/lib/utils";
import { markAsRead, toggleBookmarkStatus, deleteNotification } from "@/lib/actions";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogCancel,
  AlertDialogAction,
} from "@/components/ui/alert-dialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function NotificationCard({ notification }: { notification: Notification }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleOpenChange = async (open: boolean) => {
    setIsOpen(open);

    // If we're opening the dialog and it's unread, mark it as read on the backend.
    if (open && !notification.read) {
      await markAsRead(notification.id);
    }
    
    // If we're closing the dialog, refresh the server data to re-sort the list.
    if (!open) {
      router.refresh();
    }
  };

  const cardWrapperClasses = cn(
    "p-0.5 rounded-lg transition-all duration-300",
    !notification.read
      ? "bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/10"
      : "bg-border/50"
  );

  // Use a more predictable date format for hydration
  const createdAtDate = new Date(notification.createdAt);
  const timeDisplay = isClient 
    ? formatDistanceToNow(createdAtDate, { addSuffix: true })
    : format(createdAtDate, "MMM dd, yyyy");

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <div className={cardWrapperClasses}>
        <div className="bg-card rounded-md group relative">
          <DialogTrigger asChild>
            <div className="p-4 cursor-pointer">
              <div className="flex items-start text-left w-full space-x-4 pr-20">
                <div className="flex-shrink-0 pt-1">
                  <Tag className="h-4 w-4 text-accent" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-base text-foreground">
                    {notification.headline}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                    {notification.summary}
                  </p>
                  <div className="flex items-center gap-4 mt-3">
                    <Badge variant={notification.read ? "secondary" : "default"} className={cn(notification.read ? "" : "bg-primary/20 text-primary border border-primary/50", "capitalize")}>
                      {notification.type}
                    </Badge>
                     <p className="text-xs text-muted-foreground">
                      {timeDisplay}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </DialogTrigger>
          
          <div className="absolute top-4 right-4 flex items-center space-x-0">
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  aria-label={notification.bookmarked ? 'Remove bookmark' : 'Add bookmark'}
                >
                  <Bookmark
                    className={cn(
                      "h-4 w-4",
                      notification.bookmarked && "fill-primary text-primary"
                    )}
                  />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    {notification.bookmarked
                      ? 'Remove bookmark?'
                      : 'Add bookmark?'}
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    {notification.bookmarked
                      ? 'This will remove the bookmark from this notification.'
                      : 'This will bookmark this notification.'}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <form action={toggleBookmarkStatus}>
                    <input type="hidden" name="id" value={notification.id} />
                    <AlertDialogAction type="submit">Confirm</AlertDialogAction>
                  </form>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
                  aria-label="Delete notification"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you sure you want to delete this?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <form action={deleteNotification}>
                    <input type="hidden" name="id" value={notification.id} />
                    <AlertDialogAction
                      type="submit"
                      className="bg-destructive text-destructive-foreground hover:bg-destructive/80"
                    >
                      Delete
                    </AlertDialogAction>
                  </form>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </div>
      <DialogContent className="sm:max-w-2xl bg-card border-primary/50">
        <DialogHeader>
          <DialogTitle className="font-headline text-2xl">{notification.headline}</DialogTitle>
          <DialogDescription asChild>
            <div className="flex items-center gap-4 mt-1">
              <Badge variant={"secondary"} className="capitalize">{notification.type}</Badge>
              <p className="text-sm text-muted-foreground">{format(createdAtDate, "PPP p")}</p>
            </div>
          </DialogDescription>
        </DialogHeader>
        <div className="prose prose-sm dark:prose-invert max-w-none py-4 text-foreground/90 max-h-[60vh] overflow-y-auto pr-4">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {notification.content}
            </ReactMarkdown>
        </div>
        <DialogFooter>
            <Button onClick={() => handleOpenChange(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

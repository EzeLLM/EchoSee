export interface Notification {
  id: string;
  headline: string;
  content: string;
  summary: string;
  type: string;
  read: boolean;
  bookmarked: boolean;
  createdAt: Date;
}

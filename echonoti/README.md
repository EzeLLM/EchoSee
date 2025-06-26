# EchoNoti Progressive Web App

EchoNoti is a small PWA used to display notifications from external services. Notifications are stored in a lightweight SQLite database and can be accessed through simple HTTP endpoints.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:9003` by default.

## REST API

Notifications can be managed via HTTP calls. All endpoints return JSON.

### List notifications

```
GET /api/notifications
```
Returns an array of all notifications ordered by creation date.

### Create a notification

```
POST /api/notifications
Content-Type: application/json

{
  "headline": "System Update",
  "summary": "Short description",
  "content": "Full markdown content",
  "type": "System"
}
```
The response contains the created notification object.

### Get a single notification

```
GET /api/notifications/{id}
```
Returns the notification with the given `id` or `404` if it does not exist.

### Update a notification

```
PUT /api/notifications/{id}
Content-Type: application/json

{
  "headline": "New headline",
  "read": true,
  "bookmarked": true
}
```
Only the provided fields are updated. The response contains the updated notification.

### Delete a notification

```
DELETE /api/notifications/{id}
```
Removes the notification. The response will be `{ "success": true }` on success or `404` if the item is not found.

## License

This project is licensed under the MIT License.

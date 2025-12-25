import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import 'custom_button.dart';

class ChatWindow extends StatelessWidget {
  final VoidCallback onClose;
  final TextEditingController messageController;
  final VoidCallback onSendMessage;

  const ChatWindow({
    super.key,
    required this.onClose,
    required this.messageController,
    required this.onSendMessage,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AppTheme.darkGreen,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.5),
            blurRadius: 15,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: Column(
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'chat',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w300,
                  letterSpacing: 2,
                ),
              ),
              CustomButton(
                text: 'close',
                onPressed: onClose,
              ),
            ],
          ),
          const SizedBox(height: 24),
          // Messages area
          Expanded(
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                border: Border.all(color: AppTheme.sage),
              ),
              padding: const EdgeInsets.all(24),
              child: const SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Messages will appear here',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w300,
                      ),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'This is a sample message to demonstrate the width and spacing of the chat window.',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w300,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 24),
          // Input area
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: messageController,
                  decoration: AppTheme.inputDecoration.copyWith(
                    hintText: 'type a message...',
                    hintStyle: TextStyle(
                      color: AppTheme.cream.withOpacity(0.4),
                      letterSpacing: 1,
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 24,
                      vertical: 16,
                    ),
                  ),
                  style: const TextStyle(
                    color: AppTheme.cream,
                    fontWeight: FontWeight.w300,
                    fontSize: 16,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              CustomButton(
                text: 'send',
                onPressed: onSendMessage,
              ),
            ],
          ),
        ],
      ),
    );
  }
} 
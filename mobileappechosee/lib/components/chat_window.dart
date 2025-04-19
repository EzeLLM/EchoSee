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
              decoration: BoxDecoration(
                border: Border.all(color: AppTheme.sage),
              ),
              padding: const EdgeInsets.all(20),
              child: const SingleChildScrollView(
                child: Text('Messages will appear here'),
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
                  ),
                  style: const TextStyle(
                    color: AppTheme.cream,
                    fontWeight: FontWeight.w300,
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
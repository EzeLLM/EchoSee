import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import 'custom_button.dart';

class HistoryPanel extends StatelessWidget {
  final VoidCallback onClose;

  const HistoryPanel({
    super.key,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: AppTheme.darkGreen,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.5),
            blurRadius: 15,
            offset: const Offset(5, 0),
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
                'history',
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
          const SizedBox(height: 40),
          // History items
          Expanded(
            child: ListView(
              children: [
                _buildHistoryItem('conversation 1'),
                _buildHistoryItem('conversation 2'),
                _buildHistoryItem('conversation 3'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryItem(String text) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        border: Border.all(color: AppTheme.sage),
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontWeight: FontWeight.w300,
        ),
      ),
    );
  }
} 
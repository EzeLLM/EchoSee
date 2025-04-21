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
        crossAxisAlignment: CrossAxisAlignment.stretch,
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
                _buildHistoryButton('conversation 1'),
                const SizedBox(height: 16),
                _buildHistoryButton('conversation 2'),
                const SizedBox(height: 16),
                _buildHistoryButton('conversation 3'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryButton(String text) {
    return CustomButton(
      text: text,
      onPressed: () {
        // Handle history item press
      },
    );
  }
} 
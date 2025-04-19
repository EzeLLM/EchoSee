import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import 'custom_button.dart';

class SettingsMenu extends StatelessWidget {
  final VoidCallback onClose;

  const SettingsMenu({
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
            offset: const Offset(-5, 0),
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
                'settings',
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
          // Settings options
          _buildSettingsOption('theme'),
          _buildSettingsOption('notifications'),
          _buildSettingsOption('text size'),
          _buildSettingsOption('language'),
        ],
      ),
    );
  }

  Widget _buildSettingsOption(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: AppTheme.sage.withOpacity(0.3),
          ),
        ),
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontWeight: FontWeight.w300,
          letterSpacing: 1,
        ),
      ),
    );
  }
} 
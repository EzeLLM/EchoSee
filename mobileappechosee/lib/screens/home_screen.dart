import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../components/custom_button.dart';
import '../components/chat_window.dart';
import '../components/settings_menu.dart';
import '../components/history_panel.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _messageController = TextEditingController();
  bool _isChatOpen = false;
  bool _isSettingsOpen = false;
  bool _isHistoryOpen = false;
  String _currentPersona = 'Persona';
  int _currentPersonaIndex = 0;
  final List<String> _personas = ['Casual', 'Professional', 'Creative', 'Technical'];

  void _toggleChat() {
    setState(() {
      _isChatOpen = !_isChatOpen;
      _isSettingsOpen = false;
      _isHistoryOpen = false;
    });
  }

  void _toggleSettings() {
    setState(() {
      _isSettingsOpen = !_isSettingsOpen;
      _isChatOpen = false;
      _isHistoryOpen = false;
    });
  }

  void _toggleHistory() {
    setState(() {
      _isHistoryOpen = !_isHistoryOpen;
      _isChatOpen = false;
      _isSettingsOpen = false;
    });
  }

  void _cyclePersona() {
    setState(() {
      _currentPersonaIndex = (_currentPersonaIndex + 1) % _personas.length;
      _currentPersona = _personas[_currentPersonaIndex];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkGreen,
      body: SafeArea(
        child: Stack(
          children: [
            // Main content
            Column(
              children: [
                const SizedBox(height: 32), // Added top spacing
                // Text field container
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: TextField(
                      maxLines: null,
                      expands: true,
                      textAlignVertical: TextAlignVertical.top,
                      decoration: AppTheme.inputDecoration.copyWith(
                        hintText: 'Enter your message here...',
                        hintStyle: TextStyle(
                          color: AppTheme.cream.withOpacity(0.4),
                          letterSpacing: 1,
                        ),
                      ),
                      style: const TextStyle(
                        color: AppTheme.cream,
                        fontWeight: FontWeight.w300,
                        fontSize: 16,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 32), // Added spacing between text field and buttons
                // Buttons container
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: Column(
                      children: [
                        Expanded(
                          child: Row(
                            children: [
                              Expanded(
                                child: CustomButton(
                                  text: _currentPersona,
                                  onPressed: _cyclePersona,
                                  isPersonaButton: true,
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: CustomButton(
                                  text: 'Chat',
                                  onPressed: _toggleChat,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        Expanded(
                          child: Row(
                            children: [
                              Expanded(
                                child: CustomButton(
                                  text: 'Settings',
                                  onPressed: _toggleSettings,
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: CustomButton(
                                  text: 'History',
                                  onPressed: _toggleHistory,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 32), // Added bottom spacing
              ],
            ),
            // Overlay
            if (_isChatOpen || _isSettingsOpen || _isHistoryOpen)
              GestureDetector(
                onTap: () {
                  setState(() {
                    _isChatOpen = false;
                    _isSettingsOpen = false;
                    _isHistoryOpen = false;
                  });
                },
                child: Container(
                  color: AppTheme.darkGreen.withOpacity(0.9),
                ),
              ),
            // Chat window
            if (_isChatOpen)
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                height: MediaQuery.of(context).size.height * 0.8,
                child: ChatWindow(
                  onClose: _toggleChat,
                  messageController: _messageController,
                  onSendMessage: () {
                    // Handle send message
                  },
                ),
              ),
            // Settings menu
            if (_isSettingsOpen)
              Positioned(
                top: 0,
                right: 0,
                bottom: 0,
                left: 0,
                child: SettingsMenu(
                  onClose: _toggleSettings,
                ),
              ),
            // History panel
            if (_isHistoryOpen)
              Positioned(
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                child: HistoryPanel(
                  onClose: _toggleHistory,
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    super.dispose();
  }
} 
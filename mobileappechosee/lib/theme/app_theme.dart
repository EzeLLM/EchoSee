import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const darkGreen = Color(0xFF0F100C);
  static const darkGray = Color(0xFF282922);
  static const sage = Color(0xFF4D5449);
  static const cream = Color(0xFFDCCFBD);

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.dark(
        primary: cream,
        secondary: sage,
        background: darkGreen,
        surface: darkGreen,
      ),
      textTheme: GoogleFonts.spaceGroteskTextTheme(
        ThemeData.dark().textTheme,
      ).apply(
        bodyColor: cream,
        displayColor: cream,
      ),
      scaffoldBackgroundColor: darkGreen,
      appBarTheme: const AppBarTheme(
        backgroundColor: darkGreen,
        elevation: 0,
      ),
    );
  }

  static const buttonStyle = ButtonStyle(
    backgroundColor: MaterialStatePropertyAll(Colors.transparent),
    foregroundColor: MaterialStatePropertyAll(cream),
    side: MaterialStatePropertyAll(BorderSide(color: sage)),
    padding: MaterialStatePropertyAll(EdgeInsets.all(16)),
    textStyle: MaterialStatePropertyAll(
      TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.w300,
        letterSpacing: 2,
      ),
    ),
  );

  static const inputDecoration = InputDecoration(
    border: OutlineInputBorder(
      borderSide: BorderSide(color: sage),
    ),
    enabledBorder: OutlineInputBorder(
      borderSide: BorderSide(color: sage),
    ),
    focusedBorder: OutlineInputBorder(
      borderSide: BorderSide(color: cream),
    ),
    filled: true,
    fillColor: darkGreen,
    contentPadding: EdgeInsets.all(24),
  );
} 
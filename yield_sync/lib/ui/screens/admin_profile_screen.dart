import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';

class AdminProfileScreen extends StatelessWidget {
  const AdminProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        title: const Text("Admin Profile"),
        backgroundColor: AppColors.darkGreen,
      ),
      body: const Center(
        child: Text(
          "Admin Profile Screen",
          style: TextStyle(fontWeight: FontWeight.w800),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/nav.dart';
import '../../services/auth_service.dart';
import '../widgets/app_text_field.dart';
import '../widgets/primary_button.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  bool _obscure = true;
  bool _loading = false;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _loading = true);

    try {
      await AuthService().login(
        email: _emailCtrl.text,
        password: _passCtrl.text,
      );

      final role = await AuthService().getCurrentUserType();
      if (!mounted) return;

      if (role == "farmer") {
        Nav.clearAndGo(context, AppRoutes.home);
      } else if (role == "labour") {
        Nav.clearAndGo(context, AppRoutes.labourProfile);
      } else if (role == "equipment") {
        Nav.clearAndGo(context, AppRoutes.equipmentProfile); // ✅ NEW
      } else if (role == "seller") {
        // ✅ keep for old users (if any)
        Nav.clearAndGo(context, AppRoutes.sellerProfile);
      } else if (role == "admin") {
        Nav.clearAndGo(context, AppRoutes.adminProfile);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("User role not found. Contact admin.")),
        );
      }
    } on FirebaseAuthException catch (e) {
      final msg = switch (e.code) {
        'user-not-found' => "No account found for this email.",
        'wrong-password' => "Wrong password.",
        'invalid-email' => "Invalid email.",
        'too-many-requests' => "Too many attempts. Try again later.",
        _ => e.message ?? "Login failed.",
      };

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Stack(
          children: [
            Container(
              height: size.height * 0.34,
              width: double.infinity,
              decoration: const BoxDecoration(
                gradient: AppColors.heroGradient,
              ),
            ),
            ListView(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Nav.back(context),
                      icon: const Icon(Icons.arrow_back_ios_new_rounded),
                      color: Colors.white,
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      "Login",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  "Welcome back 👋",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.95),
                    fontSize: 28,
                    fontWeight: FontWeight.w900,
                    height: 1.1,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  "Sign in to continue to YieldSync",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.78),
                    fontSize: 14.5,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 22),
                Container(
                  padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: AppColors.border),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 26,
                        offset: const Offset(0, 16),
                      ),
                    ],
                  ),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "Enter your details",
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark,
                          ),
                        ),
                        const SizedBox(height: 12),
                        AppTextField(
                          controller: _emailCtrl,
                          label: "Email",
                          hint: "name@gmail.com",
                          keyboardType: TextInputType.emailAddress,
                          validator: (v) {
                            if (v == null || v.trim().isEmpty) {
                              return "Email is required";
                            }
                            final ok = RegExp(r'^[^@]+@[^@]+\.[^@]+$')
                                .hasMatch(v.trim());
                            if (!ok) return "Enter a valid email";
                            return null;
                          },
                          suffix: const Icon(Icons.email_outlined),
                        ),
                        const SizedBox(height: 12),
                        AppTextField(
                          controller: _passCtrl,
                          label: "Password",
                          hint: "Enter your password",
                          obscure: _obscure,
                          validator: (v) {
                            if (v == null || v.isEmpty) {
                              return "Password is required";
                            }
                            if (v.length < 6) return "Minimum 6 characters";
                            return null;
                          },
                          suffix: IconButton(
                            onPressed: () =>
                                setState(() => _obscure = !_obscure),
                            icon: Icon(
                              _obscure
                                  ? Icons.visibility_off
                                  : Icons.visibility,
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),
                        Align(
                          alignment: Alignment.centerRight,
                          child: TextButton(
                            onPressed: () {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text("Forgot password (next step)."),
                                ),
                              );
                            },
                            child: Text(
                              "Forgot password?",
                              style: TextStyle(
                                color: AppColors.darkGreen.withOpacity(0.75),
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 6),
                        PrimaryButton(
                          text: _loading ? "Logging in..." : "Login",
                          icon: Icons.login_rounded,
                          onPressed: _loading ? null : _login,
                        ),
                        const SizedBox(height: 14),
                        Row(
                          children: [
                            Expanded(child: Divider(color: AppColors.border)),
                            const SizedBox(width: 10),
                            Text(
                              "or",
                              style: TextStyle(
                                color: AppColors.textDark.withOpacity(0.55),
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(child: Divider(color: AppColors.border)),
                          ],
                        ),
                        const SizedBox(height: 14),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              "Don't have an account?",
                              style: TextStyle(
                                color: AppColors.textDark.withOpacity(0.75),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            TextButton(
                              onPressed: () =>
                                  Nav.go(context, AppRoutes.register),
                              child: const Text(
                                "Register",
                                style: TextStyle(fontWeight: FontWeight.w900),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 18),
                ClipRRect(
                  borderRadius: BorderRadius.circular(22),
                  child: Container(
                    height: 180,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      border: Border.all(color: AppColors.border),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.05),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.asset(
                          'assets/images/loginpageimage.webp',
                          fit: BoxFit.cover,
                        ),
                        Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                Colors.black.withOpacity(0.08),
                                Colors.black.withOpacity(0.28),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

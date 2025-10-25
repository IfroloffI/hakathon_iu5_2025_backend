// eslint.config.mjs
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import nestPlugin from '@nestjs/eslint-plugin';

export default tseslint.config(
  {
    languageOptions: {
      parserOptions: {
        project: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  ...tseslint.configs.stylistic,
  {
    plugins: {
      '@nestjs': nestPlugin,
    },
    rules: {
      '@typescript-eslint/no-unused-vars': 'warn',
      '@typescript-eslint/no-explicit-any': 'off', // можно включить позже
      // Отключаем "unsafe" правила — они мешают в dev
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
    },
  },
  {
    files: ['src/**/*.ts'],
    rules: {
      '@nestjs/no-unused-dependencies': 'error',
    },
  }
);

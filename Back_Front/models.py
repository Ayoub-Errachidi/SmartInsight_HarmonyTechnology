from django.db import models

class DataOriginal(models.Model):
    table_name = models.CharField(max_length=255, unique=True)  # nom table PostgreSQL
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50)
    encoding = models.CharField(max_length=50, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.table_name

class DataNettoyer(models.Model):
    table_name_nettoyee = models.CharField(max_length=255, unique=True)  # nom table PostgreSQL nettoyée
    table_originale = models.ForeignKey(
        DataOriginal,
        on_delete=models.CASCADE,
        related_name="versions_nettoyees"  # permet d'accéder aux datasets nettoyés via original.versions_nettoyees.all()
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.table_name_nettoyee} (nettoyé de {self.table_originale.table_name})"
    
class DataTransform(models.Model):
    table_name_transformee = models.CharField(max_length=255, unique=True)  # nom table PostgreSQL transformée
    table_nettoyee = models.ForeignKey(
        DataNettoyer,
        on_delete=models.CASCADE,
        related_name="versions_transformees"  # accès via nettoyée.versions_transformees.all()
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # Nouveau champ pour la colonne cible
    target_column = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return f"{self.table_name_transformee} (transformée de {self.table_nettoyee.table_name_nettoyee})"
